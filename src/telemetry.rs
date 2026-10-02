//! Fixed-size, allocation-free inference cycle telemetry.

pub const TELEMETRY_RING_CAPACITY: usize = 128;

#[derive(Clone, Copy, Debug, Default)]
pub struct TelemetryEntry {
    pub ingress_tsc: u64,
    pub compute_end_tsc: u64,
    pub egress_tsc: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ComputeCycleSummary {
    pub frames: u64,
    pub min_cycles: u64,
    pub max_cycles: u64,
    pub avg_cycles: u64,
}

/// Retains the most recent 128 frame stamps and cumulative compute statistics.
pub struct TelemetryRing {
    entries: [TelemetryEntry; TELEMETRY_RING_CAPACITY],
    next: usize,
    frames: u64,
    compute_min: u64,
    compute_max: u64,
    compute_sum: u128,
}

impl TelemetryRing {
    pub const fn new() -> Self {
        Self {
            entries: [TelemetryEntry {
                ingress_tsc: 0,
                compute_end_tsc: 0,
                egress_tsc: 0,
            }; TELEMETRY_RING_CAPACITY],
            next: 0,
            frames: 0,
            compute_min: u64::MAX,
            compute_max: 0,
            compute_sum: 0,
        }
    }

    #[inline]
    pub fn record(&mut self, entry: TelemetryEntry) {
        self.entries[self.next] = entry;
        self.next = (self.next + 1) % TELEMETRY_RING_CAPACITY;

        let compute_cycles = entry.compute_end_tsc.saturating_sub(entry.ingress_tsc);
        self.compute_min = self.compute_min.min(compute_cycles);
        self.compute_max = self.compute_max.max(compute_cycles);
        self.compute_sum = self.compute_sum.saturating_add(compute_cycles as u128);
        self.frames = self.frames.saturating_add(1);
    }

    pub fn summary(&self) -> ComputeCycleSummary {
        if self.frames == 0 {
            return ComputeCycleSummary::default();
        }
        ComputeCycleSummary {
            frames: self.frames,
            min_cycles: self.compute_min,
            max_cycles: self.compute_max,
            avg_cycles: (self.compute_sum / self.frames as u128).min(u64::MAX as u128) as u64,
        }
    }

    pub fn latest_entries(&self) -> &[TelemetryEntry; TELEMETRY_RING_CAPACITY] {
        &self.entries
    }
}
