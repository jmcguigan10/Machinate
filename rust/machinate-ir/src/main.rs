use machinate_ir::ArchitectureSpec;
use serde_json::json;
use std::env;
use std::path::PathBuf;

fn usage() -> ! {
    eprintln!("usage: machinate-ir <validate|diff|migration-plan> <args>");
    std::process::exit(2);
}

fn load_spec(path: &PathBuf) -> Result<ArchitectureSpec, String> {
    ArchitectureSpec::from_toml_file(path).map_err(|error| error.to_string())
}

fn emit(value: serde_json::Value) {
    println!("{}", serde_json::to_string_pretty(&value).unwrap());
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        usage();
    };

    match command.as_str() {
        "validate" => {
            let Some(path) = args.next() else {
                usage();
            };
            let path = PathBuf::from(path);
            match load_spec(&path) {
                Ok(spec) => match spec.validate() {
                    Ok(()) => emit(json!({
                        "ok": true,
                        "spec": path,
                        "family": spec.model.family,
                        "task": spec.model.task,
                        "parameter_count": spec.parameter_count(),
                        "param_store_manifest": spec.param_store_manifest(),
                    })),
                    Err(error) => emit(json!({
                        "ok": false,
                        "error": error.to_string(),
                        "spec": path,
                    })),
                },
                Err(error) => emit(json!({
                    "ok": false,
                    "error": error,
                    "spec": path,
                })),
            }
        }
        "diff" => {
            let (Some(old_path), Some(new_path)) = (args.next(), args.next()) else {
                usage();
            };
            let old_path = PathBuf::from(old_path);
            let new_path = PathBuf::from(new_path);
            match (load_spec(&old_path), load_spec(&new_path)) {
                (Ok(old_spec), Ok(new_spec)) => match (old_spec.validate(), new_spec.validate()) {
                    (Ok(()), Ok(())) => emit(json!({
                        "ok": true,
                        "old_spec": old_path,
                        "new_spec": new_path,
                        "diff": old_spec.diff(&new_spec),
                    })),
                    (Err(error), _) | (_, Err(error)) => emit(json!({
                        "ok": false,
                        "error": error.to_string(),
                        "old_spec": old_path,
                        "new_spec": new_path,
                    })),
                },
                (Err(error), _) | (_, Err(error)) => emit(json!({
                    "ok": false,
                    "error": error,
                    "old_spec": old_path,
                    "new_spec": new_path,
                })),
            }
        }
        "migration-plan" => {
            let (Some(old_path), Some(new_path)) = (args.next(), args.next()) else {
                usage();
            };
            let old_path = PathBuf::from(old_path);
            let new_path = PathBuf::from(new_path);
            match (load_spec(&old_path), load_spec(&new_path)) {
                (Ok(old_spec), Ok(new_spec)) => match (old_spec.validate(), new_spec.validate()) {
                    (Ok(()), Ok(())) => emit(json!({
                        "ok": true,
                        "old_spec": old_path,
                        "new_spec": new_path,
                        "migration_plan": machinate_ir::build_migration_plan(&old_spec, &new_spec),
                    })),
                    (Err(error), _) | (_, Err(error)) => emit(json!({
                        "ok": false,
                        "error": error.to_string(),
                        "old_spec": old_path,
                        "new_spec": new_path,
                    })),
                },
                (Err(error), _) | (_, Err(error)) => emit(json!({
                    "ok": false,
                    "error": error,
                    "old_spec": old_path,
                    "new_spec": new_path,
                })),
            }
        }
        _ => usage(),
    }
}
