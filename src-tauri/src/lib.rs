use std::process::Command;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|_app| {
            // Log the working directory for debugging
            let cwd = std::env::current_dir().unwrap().join("..");
            let cwd = cwd.canonicalize().unwrap();

            // Spawn the Python backend as a child process
            std::thread::spawn(move || {
                let result = Command::new("uv")
                    .args(["run", "python", "-m", "src.main"])
                    .current_dir(cwd.join("backend"))
                    .spawn();

                match result {
                    Ok(mut child) => {
                        let _ = child.wait();
                    }
                    Err(e) => {
                        eprintln!("[tauri] failed to start backend: {e}");
                    }
                }
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
