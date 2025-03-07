# Build Docker images using docker-compose
build:
	docker-compose build

# Push Docker images to the registry using docker-compose
push:
	docker-compose push

# Pull Docker images from the registry using docker-compose
pull:
	docker-compose pull

# Run the Docker containers using docker-compose
run:
	docker-compose up -d

# Stop and remove the Docker containers using docker-compose
stop:
	docker-compose down

# Remove the Docker images using docker-compose
clean:
	docker-compose down --rmi all
