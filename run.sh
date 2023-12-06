#!/bin/bash

# Print usage information
usage() {
  echo "Usage: $0 [-n <workgroup size> -v <-Dllama2.Vector[Float4|Float8|Float16]=true>] <.bin file>"
  exit 1
}

# Execute Llama2 with TornadoVM
execute_command() {
  if [ -n "$workgroup_size" ]; then
    if [ -n "$vector_mode" ]; then 
      echo "Running Llama2 with TornadoVM: workgroup size=$workgroup_size, token file=$token_file, vector mode=$vector_mode"
      tornado --jvm=" -Ds0.t0.local.workgroup.size=$workgroup_size $vector_mode" -cp target/tornadovm-llama-gpu-1.0-SNAPSHOT.jar io.github.mikepapadim.Llama2 $token_file
    else 
      echo "Running Llama2 with TornadoVM: workgroup size=$workgroup_size, token file=$token_file"
      tornado --jvm=" -Ds0.t0.local.workgroup.size=$workgroup_size " -cp target/tornadovm-llama-gpu-1.0-SNAPSHOT.jar io.github.mikepapadim.Llama2 $token_file
    fi
  else
    if [ -n "$vector_mode" ]; then
      echo "Running Llama2 with TornadoVM: default workgroup size=64, token file=$token_file, vector mode=$vector_mode"
      tornado --jvm=" -Ds0.t0.local.workgroup.size=64 $vector_mode" -cp target/tornadovm-llama-gpu-1.0-SNAPSHOT.jar io.github.mikepapadim.Llama2 $token_file
    else
      echo "Running Llama2 with TornadoVM: default workgroup size=64, token file=$token_file"
      tornado --jvm=" -Ds0.t0.local.workgroup.size=64 " -cp target/tornadovm-llama-gpu-1.0-SNAPSHOT.jar io.github.mikepapadim.Llama2 $token_file
    fi
  fi
}

# Parse command line options to identify the workgroup size, if provided
parse_options() {
  while getopts ":n:v:" opt; do
    case $opt in
      n)
        workgroup_size="$OPTARG"
        ;;
      v)
	vector_mode="$OPTARG"
	;;
      \?)
        echo "Invalid option: -$OPTARG" >&2
        usage
        ;;
      :)
        echo "Option -$OPTARG requires an argument." >&2
        usage
        ;;
    esac
  done
}

# Main function
main() {
  # Parse command line options
  parse_options "$@"

  # Shift to get the input token file, which is a mandatory input
  shift $((OPTIND - 1))

  # Check if the token file argument was provided
  if [ $# -eq 0 ]; then
    echo "Error: Missing token file argument." >&2
    usage
  fi

  token_file="$1"

  execute_command
}

main "$@"
