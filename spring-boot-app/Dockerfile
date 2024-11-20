# Use an official OpenJDK runtime as a parent image
FROM library/openjdk:17-jdk-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application JAR file into the container
COPY target/*.jar app.jar

# Copy the application properties file into the container
COPY src/main/resources/application.properties application.properties

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run the jar file
ENTRYPOINT ["java","-jar","app.jar"]