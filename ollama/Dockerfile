# Use the official Ollama image
FROM ollama/ollama:latest

# Copy the pull-llama3.sh script into the container
COPY ./pull-llama3.sh /pull-llama3.sh

# Run the script to pull the llama3 model
ENTRYPOINT ["/bin/bash", "/pull-llama3.sh"]
