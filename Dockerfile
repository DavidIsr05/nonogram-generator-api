FROM astita/openjdk21-alpine
WORKDIR /app
COPY /build/libs/* ./

EXPOSE 8080
ENTRYPOINT ["java", "-jar", "nonogramGeneratorAPI-0.0.1-SNAPSHOT.jar"]