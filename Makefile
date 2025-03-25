# Nome da imagem
IMAGE_NAME=fraud-api

# Comandos para Docker
build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -p 8000:8000 $(IMAGE_NAME)

test-payload:
	curl -X POST http://localhost:8000/predict \
	  -H "Content-Type: application/json" \
	  -d @test_payload.json | jq

# Comando para remover imagem
clean:
	docker rmi $(IMAGE_NAME)
