import os
import sys
import signal
import platform
import argparse
import subprocess

def run_command(command, shell=False):
    """Run a system command and ensure it succeeds."""
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {e}")
        sys.exit(1)

def run_inference():
    # Try to find llama-cli in different locations
    possible_paths = [
        os.path.join("build", "bin", "llama-cli"),  # Original path
        os.path.join("/Users/mguiraud/Documents/llama.cpp/build/bin", "llama-cli"),  # llama.cpp build
        os.path.join(os.path.expanduser("~"), "Documents", "llama.cpp", "build", "bin", "llama-cli")  # Home directory
    ]
    
    main_path = None
    for path in possible_paths:
        if os.path.exists(path):
            main_path = path
            print(f"Using llama.cpp binary at: {main_path}")
            break
    
    if main_path is None:
        print("Error: Could not find llama.cpp binary. Please specify the path manually.")
        sys.exit(1)
    command = [
        f'{main_path}',
        '-m', args.model,
        '-n', str(args.n_predict),
        '-t', str(args.threads),
        '-p', args.prompt,
        '-ngl', '0',
        '-c', str(args.ctx_size),
        '--temp', str(args.temperature),
        "-b", "1",
    ]
    if args.conversation:
        command.append("-cnv")
    run_command(command)

def signal_handler(sig, frame):
    print("Ctrl+C pressed, exiting...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    # Usage: python run_inference.py -p "Microsoft Corporation is an American multinational corporation and technology company headquartered in Redmond, Washington."
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument("-m", "--model", type=str, help="Path to model file", required=False, default="models/bitnet_b1_58-3B/ggml-model-i2_s.gguf")
    parser.add_argument("-n", "--n-predict", type=int, help="Number of tokens to predict when generating text", required=False, default=128)
    parser.add_argument("-p", "--prompt", type=str, help="Prompt to generate text from", required=True)
    parser.add_argument("-t", "--threads", type=int, help="Number of threads to use", required=False, default=2)
    parser.add_argument("-c", "--ctx-size", type=int, help="Size of the prompt context", required=False, default=2048)
    parser.add_argument("-temp", "--temperature", type=float, help="Temperature, a hyperparameter that controls the randomness of the generated text", required=False, default=0.8)
    parser.add_argument("-cnv", "--conversation", action='store_true', help="Whether to enable chat mode or not (for instruct models.)")

    args = parser.parse_args()
    run_inference()