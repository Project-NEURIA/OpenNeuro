use std::process::{Child, Command};
use std::sync::Mutex;

static BACKEND: Mutex<Option<Child>> = Mutex::new(None);

fn kill_backend() {
    if let Ok(mut guard) = BACKEND.lock() {
        if let Some(ref mut child) = *guard {
            let _ = child.kill();
            let _ = child.wait();
        }
        *guard = None;
    }
}

/// Detect GPU vendor and return the uv extra name.
/// Returns Some("cu128") for NVIDIA, Some("rocm") for AMD, None for other/Mac.
fn detect_gpu_extra() -> Option<&'static str> {
    if cfg!(target_os = "macos") {
        return None; // Mac uses default PyPI torch (MPS)
    }

    match gfxinfo::active_gpu() {
        Ok(gpu) => {
            let vendor = gpu.vendor().to_lowercase();
            if vendor.contains("nvidia") {
                println!("[tauri] Detected NVIDIA GPU: {}", gpu.model());
                None // CUDA is the default group
            } else if vendor.contains("amd") || vendor.contains("ati") {
                println!("[tauri] Detected AMD GPU: {}", gpu.model());
                Some("rocm")
            } else {
                println!("[tauri] Unknown GPU vendor: {}", gpu.vendor());
                None
            }
        }
        Err(e) => {
            eprintln!("[tauri] GPU detection failed: {e}");
            None
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    ctrlc::set_handler(move || {
        kill_backend();
        std::process::exit(0);
    })
    .expect("failed to set Ctrl+C handler");

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|_app| {
            let cwd = std::env::current_dir().unwrap().join("..");
            let cwd = cwd.canonicalize().unwrap();
            let backend_dir = cwd.join("backend");

            // Detect GPU and sync dependencies
            let extra = detect_gpu_extra();
            let mut sync_args: Vec<&str> = vec!["sync"];
            if cfg!(target_os = "macos") {
                sync_args.extend(["--no-group", "cuda12"]);
            } else {
                match extra {
                    Some("rocm") => {
                        sync_args.extend(["--no-group", "cuda12", "--group", "rocm"]);
                    }
                    _ => {} // NVIDIA: cuda12 is default-group, no extra flags needed
                }
            }

            println!("[tauri] Running: uv {}", sync_args.join(" "));
            match Command::new("uv")
                .args(&sync_args)
                .current_dir(&backend_dir)
                .status()
            {
                Ok(status) if status.success() => {
                    println!("[tauri] uv sync completed successfully");
                }
                Ok(status) => {
                    eprintln!("[tauri] uv sync failed with exit code: {status}");
                }
                Err(e) => {
                    eprintln!("[tauri] failed to run uv sync: {e}");
                }
            }

            // Run backend from venv
            let python = backend_dir.join(".venv/bin/python");
            match Command::new(&python)
                .args(["-m", "src.main"])
                .current_dir(&backend_dir)
                .spawn()
            {
                Ok(child) => {
                    *BACKEND.lock().unwrap() = Some(child);
                }
                Err(e) => {
                    eprintln!("[tauri] failed to start backend: {e}");
                }
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");

    kill_backend();
}
