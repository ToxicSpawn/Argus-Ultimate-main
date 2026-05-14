//! argus_bypass_sidecar
//! ====================
//! Kernel-bypass order sidecar for Argus AI.
//!
//! Listens on a Unix domain socket for order frames from Python
//! (core/kernel_bypass_stub.py) and forwards them via:
//!   1. DPDK rte_eth_tx_burst  — when DPDK feature is compiled in
//!   2. UDP fallback            — when DPDK is unavailable
//!
//! Wire protocol (little-endian):
//!   [4B magic=0xA4670001][1B action][8B qty_fixed][8B price_fixed][32B symbol]
//!
//! Build (standard UDP):
//!   cargo build --release
//!
//! Build (DPDK):
//!   cargo build --release --features dpdk
//!   (requires dpdk-sys, huge pages, NIC bound to vfio-pci)
//!
//! Run:
//!   ARGUS_GW_HOST=127.0.0.1 ARGUS_GW_PORT=9001 ./target/release/argus_bypass_sidecar

use std::env;
use std::io::{self, Read};
use std::net::UdpSocket;
use std::os::unix::net::UnixListener;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

#[cfg(feature = "dpdk")]
mod dpdk_sender;

const MAGIC: u32 = 0xA467_0001;
const FRAME_SIZE: usize = 4 + 1 + 8 + 8 + 32; // 53 bytes
const PRICE_SCALE: f64 = 1_000_000.0;
const SOCKET_PATH: &str = "/tmp/argus_bypass.sock";

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

pub struct Stats {
    pub orders_sent: AtomicU64,
    pub orders_failed: AtomicU64,
    pub bytes_sent: AtomicU64,
    pub dpdk_sent: AtomicU64,
    pub udp_fallback_sent: AtomicU64,
    pub total_latency_ns: AtomicU64,
}

impl Stats {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            orders_sent: AtomicU64::new(0),
            orders_failed: AtomicU64::new(0),
            bytes_sent: AtomicU64::new(0),
            dpdk_sent: AtomicU64::new(0),
            udp_fallback_sent: AtomicU64::new(0),
            total_latency_ns: AtomicU64::new(0),
        })
    }

    fn avg_latency_ns(&self) -> u64 {
        let sent = self.orders_sent.load(Ordering::Relaxed);
        if sent == 0 { return 0; }
        self.total_latency_ns.load(Ordering::Relaxed) / sent
    }
}

// ---------------------------------------------------------------------------
// Order frame
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct OrderFrame {
    pub action: u8,
    pub qty: f64,
    pub price: f64,
    pub symbol: String,
}

fn parse_frame(buf: &[u8; FRAME_SIZE]) -> Option<OrderFrame> {
    let magic = u32::from_le_bytes(buf[0..4].try_into().ok()?);
    if magic != MAGIC {
        eprintln!("[sidecar] bad magic: 0x{:08X}", magic);
        return None;
    }
    let action = buf[4];
    let qty_fixed = i64::from_le_bytes(buf[5..13].try_into().ok()?);
    let price_fixed = i64::from_le_bytes(buf[13..21].try_into().ok()?);
    let sym_bytes = &buf[21..53];
    let symbol = String::from_utf8_lossy(sym_bytes)
        .trim_end_matches('\0')
        .to_string();
    Some(OrderFrame {
        action,
        qty: qty_fixed as f64 / PRICE_SCALE,
        price: price_fixed as f64 / PRICE_SCALE,
        symbol,
    })
}

fn action_str(action: u8) -> &'static str {
    match action {
        0x01 => "BUY_MKT",
        0x02 => "SELL_MKT",
        0x03 => "BUY_LMT",
        0x04 => "SELL_LMT",
        _ => "UNKNOWN",
    }
}

// ---------------------------------------------------------------------------
// FIX-lite payload builder
// ---------------------------------------------------------------------------

fn build_payload(frame: &OrderFrame) -> String {
    let ts_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "{}|{}|{:.8}|{:.6}|{}",
        action_str(frame.action), frame.symbol, frame.qty, frame.price, ts_ns
    )
}

// ---------------------------------------------------------------------------
// DPDK sender (compiled only with --features dpdk)
// ---------------------------------------------------------------------------

#[cfg(feature = "dpdk")]
mod dpdk_sender {
    use super::{OrderFrame, Stats, build_payload};
    use std::sync::Arc;
    use std::time::Instant;
    use std::sync::atomic::Ordering;

    // dpdk-sys bindings — requires dpdk >= 23.11 and NIC bound to vfio-pci
    // Hugepages: echo 1024 > /proc/sys/vm/nr_hugepages
    // NIC bind:  dpdk-devbind.py --bind=vfio-pci <PCI_ADDR>
    extern crate dpdk_sys;
    use dpdk_sys::*;

    static mut TX_MBUF_POOL: *mut rte_mempool = std::ptr::null_mut();
    static mut PORT_ID: u16 = 0;
    static mut QUEUE_ID: u16 = 0;

    /// Initialise DPDK EAL and set up TX queue.
    /// Call once before send_dpdk().
    pub unsafe fn dpdk_init(port_id: u16, queue_id: u16) -> bool {
        let args: Vec<std::ffi::CString> = vec![
            std::ffi::CString::new("argus_sidecar").unwrap(),
            std::ffi::CString::new("--proc-type=primary").unwrap(),
            std::ffi::CString::new("-l").unwrap(),
            std::ffi::CString::new("0").unwrap(),
        ];
        let mut argv: Vec<*mut i8> = args.iter()
            .map(|s| s.as_ptr() as *mut i8)
            .collect();
        let ret = rte_eal_init(argv.len() as i32, argv.as_mut_ptr());
        if ret < 0 {
            eprintln!("[dpdk] rte_eal_init failed: {}", ret);
            return false;
        }
        TX_MBUF_POOL = rte_pktmbuf_pool_create(
            b"argus_mbuf_pool\0".as_ptr() as *const i8,
            8191,    // n mbufs (power-of-2 minus 1)
            256,     // cache size
            0,       // priv size
            RTE_MBUF_DEFAULT_BUF_SIZE as u16,
            rte_socket_id() as i32,
        );
        if TX_MBUF_POOL.is_null() {
            eprintln!("[dpdk] mbuf pool creation failed");
            return false;
        }
        PORT_ID = port_id;
        QUEUE_ID = queue_id;
        println!("[dpdk] EAL init ok, port={} queue={}", port_id, queue_id);
        true
    }

    /// Send order frame via rte_eth_tx_burst (zero-copy kernel bypass).
    pub unsafe fn send_dpdk(frame: &OrderFrame, stats: &Arc<Stats>) -> bool {
        let t0 = Instant::now();
        let payload = build_payload(frame);
        let payload_bytes = payload.as_bytes();

        // Allocate mbuf
        let mbuf = rte_pktmbuf_alloc(TX_MBUF_POOL);
        if mbuf.is_null() {
            eprintln!("[dpdk] mbuf alloc failed");
            stats.orders_failed.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        // Write payload into mbuf data area
        let data_ptr = rte_pktmbuf_mtod(mbuf, *mut u8);
        std::ptr::copy_nonoverlapping(
            payload_bytes.as_ptr(),
            data_ptr,
            payload_bytes.len(),
        );
        (*mbuf).data_len = payload_bytes.len() as u16;
        (*mbuf).pkt_len = payload_bytes.len() as u32;

        // TX burst (single packet)
        let mut mbufs = [mbuf];
        let sent = rte_eth_tx_burst(PORT_ID, QUEUE_ID, mbufs.as_mut_ptr(), 1);
        let lat_ns = t0.elapsed().as_nanos() as u64;

        if sent == 1 {
            stats.dpdk_sent.fetch_add(1, Ordering::Relaxed);
            stats.orders_sent.fetch_add(1, Ordering::Relaxed);
            stats.bytes_sent.fetch_add(payload_bytes.len() as u64, Ordering::Relaxed);
            stats.total_latency_ns.fetch_add(lat_ns, Ordering::Relaxed);
            println!(
                "[dpdk] TX {} {} qty={:.8} px={:.6} lat={}ns",
                action_str(frame.action), frame.symbol,
                frame.qty, frame.price, lat_ns
            );
            true
        } else {
            // TX ring full — free mbuf and fall through to UDP
            rte_pktmbuf_free(mbuf);
            stats.orders_failed.fetch_add(1, Ordering::Relaxed);
            eprintln!("[dpdk] tx_burst returned 0 — TX ring full");
            false
        }
    }
}

// ---------------------------------------------------------------------------
// UDP fallback sender
// ---------------------------------------------------------------------------

fn send_udp(frame: &OrderFrame, udp: &UdpSocket, stats: &Arc<Stats>) {
    let payload = build_payload(frame);
    let t0 = Instant::now();
    match udp.send(payload.as_bytes()) {
        Ok(n) => {
            let lat_ns = t0.elapsed().as_nanos() as u64;
            stats.orders_sent.fetch_add(1, Ordering::Relaxed);
            stats.udp_fallback_sent.fetch_add(1, Ordering::Relaxed);
            stats.bytes_sent.fetch_add(n as u64, Ordering::Relaxed);
            stats.total_latency_ns.fetch_add(lat_ns, Ordering::Relaxed);
            println!(
                "[udp] SENT {} {} qty={:.8} px={:.6} lat={}ns",
                action_str(frame.action), frame.symbol,
                frame.qty, frame.price, lat_ns
            );
        }
        Err(e) => {
            stats.orders_failed.fetch_add(1, Ordering::Relaxed);
            eprintln!("[udp] send failed: {}", e);
        }
    }
}

// ---------------------------------------------------------------------------
// Unified send — DPDK first, UDP fallback
// ---------------------------------------------------------------------------

fn send_order(frame: &OrderFrame, udp: &UdpSocket, stats: &Arc<Stats>) {
    #[cfg(feature = "dpdk")]
    {
        let sent = unsafe { dpdk_sender::send_dpdk(frame, stats) };
        if sent { return; }
        // fall through to UDP on DPDK failure
    }
    send_udp(frame, udp, stats);
}

// ---------------------------------------------------------------------------
// Connection handler
// ---------------------------------------------------------------------------

fn handle_client(
    mut stream: std::os::unix::net::UnixStream,
    udp: UdpSocket,
    stats: Arc<Stats>,
) {
    let mut buf = [0u8; FRAME_SIZE];
    loop {
        match stream.read_exact(&mut buf) {
            Ok(_) => {
                if let Some(frame) = parse_frame(&buf) {
                    send_order(&frame, &udp, &stats);
                }
            }
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                println!("[sidecar] client disconnected");
                break;
            }
            Err(e) => {
                eprintln!("[sidecar] read error: {}", e);
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stats printer thread
// ---------------------------------------------------------------------------

fn stats_thread(stats: Arc<Stats>, interval: Duration) {
    thread::spawn(move || loop {
        thread::sleep(interval);
        println!(
            "[sidecar stats] sent={} failed={} dpdk={} udp_fallback={} avg_lat={}ns",
            stats.orders_sent.load(Ordering::Relaxed),
            stats.orders_failed.load(Ordering::Relaxed),
            stats.dpdk_sent.load(Ordering::Relaxed),
            stats.udp_fallback_sent.load(Ordering::Relaxed),
            stats.avg_latency_ns(),
        );
    });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> io::Result<()> {
    let gw_host = env::var("ARGUS_GW_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let gw_port = env::var("ARGUS_GW_PORT").unwrap_or_else(|_| "9001".into());
    let socket_path = env::var("ARGUS_BYPASS_SOCKET").unwrap_or_else(|_| SOCKET_PATH.into());
    let gw_addr = format!("{}:{}", gw_host, gw_port);

    // DPDK init (no-op when not compiled with --features dpdk)
    #[cfg(feature = "dpdk")]
    {
        let port_id: u16 = env::var("ARGUS_DPDK_PORT")
            .unwrap_or_else(|_| "0".into())
            .parse().unwrap_or(0);
        let queue_id: u16 = env::var("ARGUS_DPDK_QUEUE")
            .unwrap_or_else(|_| "0".into())
            .parse().unwrap_or(0);
        let ok = unsafe { dpdk_sender::dpdk_init(port_id, queue_id) };
        if !ok {
            eprintln!("[sidecar] DPDK init failed — falling back to UDP for all orders");
        }
    }

    // Remove stale socket
    if Path::new(&socket_path).exists() {
        std::fs::remove_file(&socket_path)?;
    }

    // UDP socket to gateway (used as fallback or primary)
    let udp = UdpSocket::bind("0.0.0.0:0")?;
    udp.connect(&gw_addr)?;
    println!("[sidecar] UDP gateway: {}", gw_addr);

    let stats = Stats::new();
    stats_thread(stats.clone(), Duration::from_secs(30));

    let listener = UnixListener::bind(&socket_path)?;
    println!("[sidecar] listening on {} (DPDK={})",
        socket_path,
        cfg!(feature = "dpdk"));

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let udp_clone = udp.try_clone()?;
                let stats_clone = stats.clone();
                thread::spawn(move || handle_client(s, udp_clone, stats_clone));
            }
            Err(e) => eprintln!("[sidecar] accept error: {}", e),
        }
    }
    Ok(())
}
