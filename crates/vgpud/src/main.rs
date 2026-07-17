//! `vgpud` binary: parse flags, serve until Ctrl-C.
//!
//! Flag parsing is hand-rolled `--key value` pairs — the same
//! zero-dependency discipline as everywhere else, and at six flags a
//! parser generator would be more code than this.

use std::net::SocketAddr;
use std::process::ExitCode;
use std::time::Duration;

use vgpu_core::node::PhysGpuConfig;
use vgpu_core::types::FRAME_SIZE;
use vgpud::{serve, AutoTick, DaemonConfig};

const USAGE: &str = "\
vgpud — VGPU-X node daemon

USAGE:
  vgpud [--listen ADDR] [--name NAME] [--vram-mib N] [--slice-cycles N]
        [--auto-tick-ms N] [--auto-tick-budget N]

FLAGS (defaults in parens):
  --listen ADDR         address to bind (127.0.0.1:7677)
  --name NAME           card name for inventory (sim-0)
  --vram-mib N          simulated VRAM in MiB (1024)
  --slice-cycles N      scheduler time-slice target (10000)
  --auto-tick-ms N      tick the GPU every N ms of idleness; 0 disables (10)
  --auto-tick-budget N  cycle budget per auto tick (100000)
";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("vgpud: {msg}");
            eprintln!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut listen: SocketAddr = "127.0.0.1:7677".parse().expect("valid default");
    let mut name = "sim-0".to_string();
    let mut vram_mib: u64 = 1024;
    let mut slice_cycles: u64 = 10_000;
    let mut auto_tick_ms: u64 = 10;
    let mut auto_tick_budget: u64 = 100_000;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            println!("{USAGE}");
            return Ok(());
        }
        let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--listen" => listen = value.parse().map_err(|e| format!("--listen: {e}"))?,
            "--name" => name = value,
            "--vram-mib" => vram_mib = value.parse().map_err(|e| format!("--vram-mib: {e}"))?,
            "--slice-cycles" => {
                slice_cycles = value.parse().map_err(|e| format!("--slice-cycles: {e}"))?
            }
            "--auto-tick-ms" => {
                auto_tick_ms = value.parse().map_err(|e| format!("--auto-tick-ms: {e}"))?
            }
            "--auto-tick-budget" => {
                auto_tick_budget = value
                    .parse()
                    .map_err(|e| format!("--auto-tick-budget: {e}"))?
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }

    let config = DaemonConfig {
        gpu: PhysGpuConfig {
            name,
            vram_bytes: vram_mib * 1024 * 1024 / FRAME_SIZE * FRAME_SIZE,
            slice_cycles,
        },
        auto_tick: (auto_tick_ms > 0).then(|| AutoTick {
            interval: Duration::from_millis(auto_tick_ms),
            budget: auto_tick_budget,
        }),
    };

    let handle = serve(listen, config).map_err(|e| format!("bind {listen}: {e}"))?;
    println!("vgpud listening on {}", handle.addr);

    // Serve until the process is killed. The threads do all the work; the
    // main thread just has to not exit.
    loop {
        std::thread::park();
    }
}
