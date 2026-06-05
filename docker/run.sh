#!/bin/bash

# Build the dockerfile
docker compose build

# Allow X server connection
xhost +local:root
docker compose run sad_vio
# Disallow X server connection
xhost -local:root
