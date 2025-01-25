# Project architecture
![arch](https://github.com/user-attachments/assets/7ef9f13d-63cf-457f-a580-37fa11ad5257)
# Instructions to Run the Project
## Prerequisites
- Docker: Ensure Docker is installed. If not, download it from Docker's official website.
- Java and Maven: Ensure Java (JDK) and Maven are installed for the Spring Boot application.
- Check Java version: java -version
- Check Maven version: mvn -v
- Git: Ensure Git is installed to clone the repository.

## Step 1: Clone the Repository
Open a terminal or command prompt.

1. Clone the repository:
```bash
git clone https://github.com/elza02/Face-Detection-System.git
```
2. Navigate to the project directory:
```bash
cd Face-Detection-System
```
## Step 2: Build and run the Docker containers
```bash
Docker compose up --build
```
## Step 3: Run the Spring Boot App
1. Navigate to the spring-boot folder:
```bash
cd ../spring-boot-app
```
2. Build the application using Maven:
```bash
mvn clean install
```
3. Run the Spring Boot application:
```bash
mvn spring-boot:run
   ```
4. The Spring Boot app will be running at: **http://localhost:8080**


