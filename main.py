import argparse
from src.inference import run_simulation

def main():
    parser = argparse.ArgumentParser(description="Ads Behavior Simulator")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    args = parser.parse_args()

    run_simulation(config_path=args.config)

if __name__ == "__main__":
    main()

