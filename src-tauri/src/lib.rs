use std::process::{Child, Command};
use std::sync::Mutex;

static BACKEND: Mutex<Option<Child>> = Mutex::new(None);

fn kill_backend() {
    if let Ok(mut guard) = BACKEND.lock() {
        if let Some(ref mut child) = *guard {
            // Kill the entire process group (uv + python subprocess)
            #[cfg(unix)]
            {
                use std::os::unix::process::CommandExt;
                unsafe {
                    let pid = child.id() as i32;
                    libc::kill(-pid, libc::SIGTERM);
                }
            }
            #[cfg(not(unix))]
            {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
        *guard = None;
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|_app| {
            let cwd = std::env::current_dir().unwrap().join("..");
            let cwd = cwd.canonicalize().unwrap();

            // Spawn backend in its own process group so we can kill the whole tree
            let mut cmd = Command::new("uv");
            cmd.args(["run", "python", "-m", "src.main"])
                .current_dir(cwd.join("backend"));

            #[cfg(unix)]
            {
                use std::os::unix::process::CommandExt;
                unsafe {
                    cmd.pre_exec(|| {
                        libc::setpgid(0, 0);
                        Ok(())
                    });
                }
            }

            match cmd.spawn() {
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
