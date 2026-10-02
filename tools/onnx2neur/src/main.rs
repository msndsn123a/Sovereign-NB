#[cfg(test)]
use ed25519_dalek::Verifier;
use ed25519_dalek::{Signer, SigningKey};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

const MAGIC: &[u8; 4] = b"NEUR";
const HEADER_BYTES: usize = 80;
const BASE_HEADER_BYTES: usize = 16;
const SEED_BYTES: usize = 32;
const MLP_L1_BYTES: usize = 32 * 16;
const MLP_L2_BYTES: usize = 16 * 8;
const MLP_PAYLOAD_BYTES: usize = MLP_L1_BYTES + MLP_L2_BYTES;
const ATTENTION_POT_BYTES: usize = 1664;

#[derive(Debug, Clone)]
struct Tensor {
    dims: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Debug)]
struct Node {
    op: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attributes: HashMap<String, i64>,
    float_attributes: HashMap<String, f32>,
}

#[derive(Debug)]
struct Graph {
    nodes: Vec<Node>,
    initializers: HashMap<String, Tensor>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Quantization {
    Ternary,
    Pot,
}

fn read_varint(bytes: &[u8], cursor: &mut usize) -> Result<u64, String> {
    let mut value = 0u64;
    for shift in (0..70).step_by(7) {
        let byte = *bytes.get(*cursor).ok_or("truncated protobuf varint")?;
        *cursor += 1;
        if shift == 63 && byte > 1 {
            return Err("protobuf varint overflow".into());
        }
        value |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err("protobuf varint overflow".into())
}

fn fields(bytes: &[u8]) -> Result<Vec<(u32, u8, Vec<u8>, u64)>, String> {
    let mut result = Vec::new();
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let key = read_varint(bytes, &mut cursor)?;
        let number = (key >> 3) as u32;
        let wire = (key & 7) as u8;
        if number == 0 {
            return Err("invalid protobuf field number 0".into());
        }
        match wire {
            0 => result.push((number, wire, Vec::new(), read_varint(bytes, &mut cursor)?)),
            1 => {
                let end = cursor.checked_add(8).ok_or("protobuf length overflow")?;
                let data = bytes.get(cursor..end).ok_or("truncated protobuf fixed64")?;
                result.push((number, wire, data.to_vec(), 0));
                cursor = end;
            }
            2 => {
                let length = read_varint(bytes, &mut cursor)? as usize;
                let end = cursor
                    .checked_add(length)
                    .ok_or("protobuf length overflow")?;
                let data = bytes.get(cursor..end).ok_or("truncated protobuf bytes")?;
                result.push((number, wire, data.to_vec(), 0));
                cursor = end;
            }
            5 => {
                let end = cursor.checked_add(4).ok_or("protobuf length overflow")?;
                let data = bytes.get(cursor..end).ok_or("truncated protobuf fixed32")?;
                result.push((number, wire, data.to_vec(), 0));
                cursor = end;
            }
            _ => return Err(format!("unsupported protobuf wire type {wire}")),
        }
    }
    Ok(result)
}

fn unpack_i64(field: &[u8], output: &mut Vec<usize>) -> Result<(), String> {
    let mut cursor = 0;
    while cursor < field.len() {
        let value = read_varint(field, &mut cursor)? as i64;
        if value < 0 {
            return Err("negative ONNX tensor dimensions are unsupported".into());
        }
        output.push(value as usize);
    }
    Ok(())
}

fn parse_tensor(bytes: &[u8]) -> Result<(String, Tensor), String> {
    let mut dims = Vec::new();
    let mut data_type = 0u64;
    let mut float_values = Vec::new();
    let mut raw_data = None;
    let mut name = None;
    for (field, wire, data, scalar) in fields(bytes)? {
        match (field, wire) {
            (1, 0) => {
                if scalar > usize::MAX as u64 {
                    return Err("ONNX dimension is too large".into());
                }
                dims.push(scalar as usize);
            }
            (1, 2) => unpack_i64(&data, &mut dims)?,
            (2, 0) => data_type = scalar,
            (4, 5) => float_values.push(f32::from_le_bytes(data.try_into().unwrap())),
            (4, 2) => {
                if data.len() % 4 != 0 {
                    return Err("malformed ONNX packed float_data".into());
                }
                float_values.extend(
                    data.chunks_exact(4)
                        .map(|v| f32::from_le_bytes(v.try_into().unwrap())),
                );
            }
            (8, 2) => name = Some(String::from_utf8(data).map_err(|_| "tensor name is not UTF-8")?),
            (9, 2) => raw_data = Some(data),
            _ => {}
        }
    }
    if data_type != 1 {
        return Err(format!(
            "initializer {} is not float32",
            name.as_deref().unwrap_or("<unnamed>")
        ));
    }
    let count = dims.iter().try_fold(1usize, |n, d| {
        n.checked_mul(*d).ok_or("tensor size overflow")
    })?;
    let values = if let Some(raw) = raw_data {
        if raw.len() != count.checked_mul(4).ok_or("tensor size overflow")? {
            return Err("ONNX raw_data length does not match tensor dimensions".into());
        }
        raw.chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect()
    } else {
        float_values
    };
    if dims.is_empty() || values.len() != count {
        return Err("ONNX initializer dimensions/data length mismatch".into());
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err("ONNX initializers must contain finite float32 weights".into());
    }
    Ok((
        name.ok_or("ONNX initializer is missing a name")?,
        Tensor { dims, values },
    ))
}

fn parse_attribute(bytes: &[u8]) -> Result<(String, i64, Option<f32>), String> {
    let mut name = None;
    let mut integer = 0;
    let mut float = None;
    for (field, wire, data, scalar) in fields(bytes)? {
        match (field, wire) {
            (1, 2) => {
                name = Some(String::from_utf8(data).map_err(|_| "attribute name is not UTF-8")?)
            }
            (2, 5) => float = Some(f32::from_le_bytes(data.try_into().unwrap())),
            (3, 0) => integer = scalar as i64,
            _ => {}
        }
    }
    Ok((
        name.ok_or("ONNX attribute is missing a name")?,
        integer,
        float,
    ))
}

fn parse_node(bytes: &[u8]) -> Result<Node, String> {
    let mut op = None;
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut attributes = HashMap::new();
    let mut float_attributes = HashMap::new();
    for (field, wire, data, _) in fields(bytes)? {
        if wire != 2 {
            continue;
        }
        match field {
            1 => inputs.push(String::from_utf8(data).map_err(|_| "node input is not UTF-8")?),
            2 => outputs.push(String::from_utf8(data).map_err(|_| "node output is not UTF-8")?),
            4 => op = Some(String::from_utf8(data).map_err(|_| "operator name is not UTF-8")?),
            5 => {
                let (name, value, float) = parse_attribute(&data)?;
                if let Some(value) = float {
                    float_attributes.insert(name.clone(), value);
                }
                attributes.insert(name, value);
            }
            _ => {}
        }
    }
    Ok(Node {
        op: op.ok_or("ONNX node is missing op_type")?,
        inputs,
        outputs,
        attributes,
        float_attributes,
    })
}

fn parse_graph(bytes: &[u8]) -> Result<Graph, String> {
    let mut nodes = Vec::new();
    let mut initializers = HashMap::new();
    for (field, wire, data, _) in fields(bytes)? {
        if wire == 2 && field == 1 {
            nodes.push(parse_node(&data)?);
        } else if wire == 2 && field == 5 {
            let (name, tensor) = parse_tensor(&data)?;
            if initializers.insert(name.clone(), tensor).is_some() {
                return Err(format!("duplicate ONNX initializer {name}"));
            }
        }
    }
    Ok(Graph {
        nodes,
        initializers,
    })
}

fn parse_model(bytes: &[u8]) -> Result<Graph, String> {
    for (field, wire, data, _) in fields(bytes)? {
        if field == 7 && wire == 2 {
            return parse_graph(&data);
        }
    }
    Err("ONNX ModelProto does not contain a graph".into())
}

fn normalized_matrix(tensor: &Tensor, transpose: bool) -> Result<(usize, usize, Vec<f32>), String> {
    if tensor.dims.len() != 2 {
        return Err("linear layer weights must be rank-2 float32 tensors".into());
    }
    let (rows, cols) = (tensor.dims[0], tensor.dims[1]);
    if transpose {
        let mut transposed = vec![0.0; tensor.values.len()];
        for row in 0..rows {
            for col in 0..cols {
                transposed[col * rows + row] = tensor.values[row * cols + col];
            }
        }
        Ok((cols, rows, transposed))
    } else {
        Ok((rows, cols, tensor.values.clone()))
    }
}

fn layer_matrix(graph: &Graph, node: &Node) -> Result<(usize, usize, Vec<f32>), String> {
    if node.op != "Gemm" && node.op != "MatMul" {
        return Err(format!(
            "unsupported ONNX operator {}; expected Gemm or MatMul",
            node.op
        ));
    }
    if node.op == "Gemm"
        && (node.attributes.get("transA").copied().unwrap_or(0) != 0
            || node
                .float_attributes
                .get("alpha")
                .is_some_and(|value| *value != 1.0)
            || node
                .float_attributes
                .get("beta")
                .is_some_and(|value| *value != 1.0))
    {
        return Err("Gemm transA, alpha, and beta must be their ONNX defaults".into());
    }
    let weight_index = 1usize;
    let weight_name = node
        .inputs
        .get(weight_index)
        .ok_or("linear node is missing weights")?;
    let tensor = graph
        .initializers
        .get(weight_name)
        .ok_or_else(|| format!("weight input {weight_name} must be a float initializer"))?;
    // MatMul weights are [input, output]; Gemm defaults to the same layout
    // unless transB explicitly asks for [output, input].
    let transpose = node.op == "MatMul" || node.attributes.get("transB").copied().unwrap_or(0) == 0;
    let matrix = normalized_matrix(tensor, transpose)?;
    if node.op == "Gemm" && node.inputs.get(2).is_some_and(|bias| !bias.is_empty()) {
        let bias = graph
            .initializers
            .get(&node.inputs[2])
            .ok_or("Gemm bias must be a constant initializer")?;
        if bias.values.iter().any(|value| value.abs() > 1e-7) {
            return Err("nonzero Gemm biases are not representable by NEUR v2".into());
        }
    }
    Ok(matrix)
}

fn quantize_ternary(value: f32, threshold: f32) -> i8 {
    if value > threshold {
        1
    } else if value < -threshold {
        -1
    } else {
        0
    }
}

fn quantize_pot(value: f32, threshold: f32) -> u8 {
    if value.abs() <= threshold || value == 0.0 {
        return 0;
    }
    let exponent = value.abs().log2().round().clamp(0.0, 6.0) as u8;
    if value.is_sign_negative() {
        9 + exponent
    } else {
        1 + exponent
    }
}

fn pack_ternary(values: &[i8], output: &mut Vec<u8>) {
    for chunk in values.chunks(4) {
        let mut byte = 0u8;
        for (lane, weight) in chunk.iter().enumerate() {
            let code = match weight {
                1 => 1,
                -1 => 3,
                _ => 0,
            };
            byte |= code << (lane * 2);
        }
        output.push(byte);
    }
}

fn matrices_to_payload(
    graph: &Graph,
    quant: Quantization,
    threshold: f32,
) -> Result<(u8, Vec<u8>), String> {
    let mut matrices = Vec::new();
    for node in &graph.nodes {
        if node.op == "Gemm" || node.op == "MatMul" {
            matrices.push((node, layer_matrix(graph, node)?));
        } else {
            return Err(format!(
                "unsupported ONNX node {}; this compiler accepts linear Gemm/MatMul graphs only",
                node.op
            ));
        }
    }
    if quant == Quantization::Ternary {
        if matrices.len() != 2
            || matrices[0].1 .0 != 32
            || matrices[0].1 .1 != 64
            || matrices[1].1 .0 != 16
            || matrices[1].1 .1 != 32
        {
            return Err(
                "ternary MLP requires two linear layers with shapes 32x64 and 16x32".into(),
            );
        }
        let first_output = matrices[0]
            .0
            .outputs
            .first()
            .ok_or("first linear node has no output")?;
        if matrices[1].0.inputs.first() != Some(first_output) {
            return Err("ternary MLP nodes must form a connected two-layer linear chain".into());
        }
        let mut payload = Vec::with_capacity(MLP_PAYLOAD_BYTES);
        for (_, (_, _, values)) in matrices {
            pack_ternary(
                &values
                    .iter()
                    .map(|v| quantize_ternary(*v, threshold))
                    .collect::<Vec<_>>(),
                &mut payload,
            );
        }
        Ok((0, payload))
    } else {
        if matrices.len() != 4 {
            return Err("PoT attention export requires four projection matrices: Q, K, V (16x64) and O (16x16)".into());
        }
        let mut qkv = Vec::new();
        let mut output = None;
        for (_, (rows, cols, values)) in matrices {
            if (rows, cols) == (16, 64) {
                qkv.extend(values);
            } else if (rows, cols) == (16, 16) && output.is_none() {
                output = Some(values);
            } else {
                return Err(format!(
                    "unsupported PoT attention projection shape {rows}x{cols}"
                ));
            }
        }
        if qkv.len() != 3 * 16 * 64 {
            return Err("PoT attention requires exactly three 16x64 projections".into());
        }
        qkv.extend(output.ok_or("PoT attention requires one 16x16 output projection")?);
        let mut payload = Vec::with_capacity(ATTENTION_POT_BYTES * 2);
        for value in qkv {
            payload.push(quantize_pot(value, threshold));
        }
        let mut packed = Vec::with_capacity(ATTENTION_POT_BYTES / 2);
        for pair in payload.chunks_exact(2) {
            packed.push(pair[0] | (pair[1] << 4));
        }
        Ok((1, packed))
    }
}

fn make_header(model_type: u8, pot_scale: u8) -> [u8; BASE_HEADER_BYTES] {
    let mut header = [0u8; BASE_HEADER_BYTES];
    header[..4].copy_from_slice(MAGIC);
    header[4..8].copy_from_slice(&2u32.to_le_bytes());
    header[8..12].copy_from_slice(&64u32.to_le_bytes());
    header[12] = model_type;
    let output_meta = 16u16
        | if model_type == 1 {
            (pot_scale as u16) << 12
        } else {
            0
        };
    header[13..15].copy_from_slice(&output_meta.to_le_bytes());
    header[15] = if model_type == 0 { 32 } else { 16 };
    if model_type == 1 {
        header[15] |= 0x40;
    }
    header
}

fn signed_shard(
    model_type: u8,
    payload: &[u8],
    block_size: usize,
    key: &SigningKey,
    pot_scale: u8,
) -> Result<Vec<u8>, String> {
    if !(512..=4096).contains(&block_size) || !block_size.is_power_of_two() {
        return Err("block size must be a power of two from 512 through 4096".into());
    }
    if model_type == 1 && pot_scale > 6 {
        return Err("PoT scale must be from 0 through 6".into());
    }
    let mut message = Vec::with_capacity(BASE_HEADER_BYTES + payload.len());
    let header = make_header(model_type, if model_type == 1 { pot_scale } else { 0 });
    message.extend_from_slice(&header);
    message.extend_from_slice(payload);
    let signature = key.sign(&message).to_bytes();
    let total = HEADER_BYTES + payload.len();
    let padded = total.div_ceil(block_size) * block_size;
    let mut shard = vec![0u8; padded];
    shard[..BASE_HEADER_BYTES].copy_from_slice(&header);
    shard[BASE_HEADER_BYTES..HEADER_BYTES].copy_from_slice(&signature);
    shard[HEADER_BYTES..total].copy_from_slice(payload);
    Ok(shard)
}

fn read_key(path: &str) -> Result<SigningKey, String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read signing seed: {error}"))?;
    if bytes.len() != SEED_BYTES {
        return Err("Ed25519 signing key file must contain exactly 32 raw seed bytes".into());
    }
    let seed: [u8; SEED_BYTES] = bytes.try_into().unwrap();
    Ok(SigningKey::from_bytes(&seed))
}

fn parse_args() -> Result<(String, String, String, Quantization, usize, f32, u8), String> {
    let mut args = env::args().skip(1);
    let mut input = None;
    let mut output = "dist/compiled.neur".to_string();
    let mut key = None;
    let mut quant = "ternary".to_string();
    let mut block_size = 512usize;
    let mut threshold = 0.25f32;
    let mut pot_scale = 0u8;
    while let Some(arg) = args.next() {
        let value = |args: &mut std::iter::Skip<std::env::Args>| {
            args.next().ok_or_else(|| format!("{arg} requires a value"))
        };
        match arg.as_str() {
            "-i" | "--input" => input = Some(value(&mut args)?),
            "-o" | "--output" => output = value(&mut args)?,
            "--key-file" => key = Some(value(&mut args)?),
            "--quant" => quant = value(&mut args)?,
            "--block-size" => block_size = value(&mut args)?.parse().map_err(|_| "invalid block size")?,
            "--threshold" => threshold = value(&mut args)?.parse().map_err(|_| "invalid quantization threshold")?,
            "--pot-scale" => pot_scale = value(&mut args)?.parse().map_err(|_| "invalid PoT scale")?,
            "-h" | "--help" => return Err("usage: onnx2neur --input model.onnx --output model.neur --key-file <32-byte-seed> [--quant ternary|pot] [--threshold 0.25] [--pot-scale 0] [--block-size 512]".into()),
            _ => return Err(format!("unknown argument {arg}")),
        }
    }
    if !threshold.is_finite() || threshold < 0.0 {
        return Err("threshold must be finite and non-negative".into());
    }
    let quant = match quant.as_str() {
        "ternary" => Quantization::Ternary,
        "pot" => Quantization::Pot,
        _ => return Err("--quant must be ternary or pot".into()),
    };
    Ok((
        input.ok_or("--input is required")?,
        output,
        key.ok_or("--key-file is required")?,
        quant,
        block_size,
        threshold,
        pot_scale,
    ))
}

fn compile(
    input: &[u8],
    quant: Quantization,
    threshold: f32,
    block_size: usize,
    key: &SigningKey,
    pot_scale: u8,
) -> Result<Vec<u8>, String> {
    let graph = parse_model(input)?;
    let (model_type, payload) = matrices_to_payload(&graph, quant, threshold)?;
    signed_shard(model_type, &payload, block_size, key, pot_scale)
}

fn run() -> Result<(), String> {
    let (input_path, output_path, key_path, quant, block_size, threshold, pot_scale) =
        parse_args()?;
    let input =
        fs::read(&input_path).map_err(|error| format!("cannot read ONNX model: {error}"))?;
    let key = read_key(&key_path)?;
    let shard = compile(&input, quant, threshold, block_size, &key, pot_scale)?;
    if let Some(parent) = Path::new(&output_path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create output directory: {error}"))?;
        }
    }
    fs::write(&output_path, &shard)
        .map_err(|error| format!("cannot write signed shard: {error}"))?;
    println!(
        "[ONNX2NEUR]: graph converted and signed as NEUR v2 ({} bytes, Ed25519)",
        shard.len()
    );
    println!(
        "[ONNX2NEUR]: model_type={} quant={:?} threshold={} block_size={}",
        shard[12], quant, threshold, block_size
    );
    println!(
        "[ONNX2NEUR]: public_key={:02x?}",
        key.verifying_key().to_bytes()
    );
    println!("[ONNX2NEUR]: output={output_path}");
    Ok(())
}

fn main() {
    if env::args().any(|arg| arg == "-h" || arg == "--help") {
        println!("usage: onnx2neur --input model.onnx --output model.neur --key-file <32-byte-seed> [--quant ternary|pot] [--threshold 0.25] [--pot-scale 0] [--block-size 512]");
        return;
    }
    if let Err(error) = run() {
        eprintln!("onnx2neur: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_varint(mut value: u64, out: &mut Vec<u8>) {
        while value >= 0x80 {
            out.push((value as u8) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }
    fn field_bytes(number: u32, bytes: &[u8], out: &mut Vec<u8>) {
        put_varint(((number as u64) << 3) | 2, out);
        put_varint(bytes.len() as u64, out);
        out.extend_from_slice(bytes);
    }
    fn field_varint(number: u32, value: u64, out: &mut Vec<u8>) {
        put_varint((number as u64) << 3, out);
        put_varint(value, out);
    }
    fn onnx_tensor(name: &str, dims: &[usize], values: &[f32]) -> Vec<u8> {
        let mut tensor = Vec::new();
        let mut packed_dims = Vec::new();
        for dim in dims {
            put_varint(*dim as u64, &mut packed_dims);
        }
        field_bytes(1, &packed_dims, &mut tensor);
        field_varint(2, 1, &mut tensor);
        for value in values {
            field_bytes(4, &value.to_le_bytes(), &mut tensor);
        }
        field_bytes(8, name.as_bytes(), &mut tensor);
        tensor
    }
    fn onnx_node(op: &str, input: &str, weight: &str, output: &str) -> Vec<u8> {
        let mut node = Vec::new();
        field_bytes(1, input.as_bytes(), &mut node);
        field_bytes(1, weight.as_bytes(), &mut node);
        field_bytes(2, output.as_bytes(), &mut node);
        field_bytes(4, op.as_bytes(), &mut node);
        if op == "Gemm" {
            let mut attribute = Vec::new();
            field_bytes(1, b"transB", &mut attribute);
            field_varint(3, 1, &mut attribute);
            field_bytes(5, &attribute, &mut node);
        }
        node
    }
    fn reference_exported_graph() -> Vec<u8> {
        let mut graph = Vec::new();
        let first: Vec<f32> = (0..32 * 64)
            .map(|i| {
                if i % 3 == 0 {
                    0.8
                } else if i % 3 == 1 {
                    -0.9
                } else {
                    0.05
                }
            })
            .collect();
        let second: Vec<f32> = (0..16 * 32)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        field_bytes(1, &onnx_node("Gemm", "input", "w1", "hidden"), &mut graph);
        field_bytes(
            1,
            &onnx_node("MatMul", "hidden", "w2", "output"),
            &mut graph,
        );
        field_bytes(5, &onnx_tensor("w1", &[32, 64], &first), &mut graph);
        field_bytes(5, &onnx_tensor("w2", &[32, 16], &second), &mut graph);
        let mut model = Vec::new();
        field_bytes(7, &graph, &mut model);
        model
    }

    #[test]
    fn exported_reference_graph_roundtrips_to_authenticated_neur_v2() {
        let seed = [7u8; 32];
        let signing_key = SigningKey::from_bytes(&seed);
        let shard = compile(
            &reference_exported_graph(),
            Quantization::Ternary,
            0.25,
            512,
            &signing_key,
            0,
        )
        .unwrap();
        assert_eq!(&shard[..4], MAGIC);
        assert_eq!(u32::from_le_bytes(shard[4..8].try_into().unwrap()), 2);
        assert_eq!(shard[12], 0);
        assert_eq!(shard.len(), 1024);
        let mut message = shard[..BASE_HEADER_BYTES].to_vec();
        message.extend_from_slice(&shard[HEADER_BYTES..HEADER_BYTES + MLP_PAYLOAD_BYTES]);
        let signature =
            ed25519_dalek::Signature::from_slice(&shard[BASE_HEADER_BYTES..HEADER_BYTES]).unwrap();
        signing_key
            .verifying_key()
            .verify(&message, &signature)
            .unwrap();
        let first_code = shard[HEADER_BYTES] & 3;
        assert_eq!(first_code, 1);
        let second_code = (shard[HEADER_BYTES] >> 2) & 3;
        assert_eq!(second_code, 3);
    }

    #[test]
    fn gemm_transpose_is_normalized() {
        let graph = parse_model(&reference_exported_graph()).unwrap();
        assert_eq!(layer_matrix(&graph, &graph.nodes[0]).unwrap().0, 32);
    }
}
