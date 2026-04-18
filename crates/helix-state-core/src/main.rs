use clap::{Parser, Subcommand};
use _helix_state_core::{pack_hlx_bundle, unpack_hlx_bundle, verify_hlx_session};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "helix-state-core")]
#[command(about = "Pack, unpack, and verify HeliX .hlx session bundles")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Pack {
        #[arg(long)]
        staging_dir: PathBuf,
        #[arg(long)]
        session_json: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
    Unpack {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        output_staging: PathBuf,
    },
    Verify {
        #[arg(long)]
        session_dir: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Pack {
            staging_dir,
            session_json,
            output_dir,
        } => {
            let receipt = pack_hlx_bundle(&staging_dir, &session_json, &output_dir)?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }
        Command::Unpack {
            bundle,
            output_staging,
        } => {
            let manifest = unpack_hlx_bundle(&bundle, &output_staging)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        Command::Verify { session_dir } => {
            let receipt = verify_hlx_session(&session_dir)?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }
    }
    Ok(())
}
