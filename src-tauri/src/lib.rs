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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    ctrlc::set_handler(move || {
        kill_backend();
        std::process::exit(0);
    })
    .expect("failed to set Ctrl+C handler");

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|_app| {
            let cwd = std::env::current_dir().unwrap().join("..");
            let cwd = cwd.canonicalize().unwrap();
            let backend_dir = cwd.join("backend");

            // Run python directly from venv — no uv middleman
            // This ensures kill() actually kills the python process
            let python = backend_dir.join(".venv/bin/python");
            match Command::new(&python)
                .args(["-m", "src.main"])
                .current_dir(&backend_dir)
                .spawn()
            {
                Ok(child) => {
                    *BACKEND.lock().unwrap() = Some(child);
                }
                Err(_) => {
                    // Fallback to uv if venv doesn't exist
                    match Command::new("uv")
                        .args(["run", "python", "-m", "src.main"])
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
                }
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");

    kill_backend();
}
