.PHONY: build run test clean deps

build:
	go build -o yesman main.go
	go build -o test_app test_app.go

deps:
	go mod download
	go mod tidy

run: build
	./yesman ./test_app

test: build
	@echo "Testing YesMan with test app..."
	@echo "Make sure to set OPENAI_API_KEY environment variable"
	./yesman ./test_app

clean:
	rm -f yesman test_app

install: build
	cp yesman ~/go/bin/ || cp yesman /usr/local/bin/