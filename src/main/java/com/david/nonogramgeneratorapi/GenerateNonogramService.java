package com.david.nonogramgeneratorapi;

import com.david.nonogramgeneratorapi.dtos.*;
import jakarta.annotation.PostConstruct;
import org.apache.commons.io.FileUtils;
import org.imgscalr.Scalr;
import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Base64;

@Service
public class GenerateNonogramService {

    private Net net;
    private final String outputPath = "src/main/resources/";
    private final String modelPath = "/Users/david/Documents/nonogramGeneratorAPI/src/main/resources/u2net.onnx";

    static {
        try {
            System.load("/Users/david/Documents/nonogramGeneratorAPI/src/main/resources/libopencv_java4120.dylib");
            System.out.println("✅");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("❌" + e.getMessage());
            System.exit(1);
        }
    }

    @PostConstruct
    public void initModel() {
        try {
            System.out.println("Loading U2Net Model from: " + modelPath);
            this.net = Dnn.readNetFromONNX(modelPath);
            if (this.net.empty()) {
                System.err.println("failed to load U2Net model");
            } else {
                System.out.println("loaded successfully.");
            }
        } catch (Exception e) {
            System.err.println("Exception loading model: " + e.getMessage());
        }
    }

    public nonogramResponseDto generateNonogram(nonogramGenerationRequestDto body) throws IOException {

        byte[] decodedBytes = Base64.getDecoder().decode(body.getImageBase64());
        File originalImageFile = new File(outputPath + "original-image.jpg");
        FileUtils.writeByteArrayToFile(originalImageFile, decodedBytes);

        String maskedImagePath = outputPath + "masked-image.png";
        applySaliencyMask(originalImageFile.getAbsolutePath(), maskedImagePath);

        File processedFile = new File(maskedImagePath);
        BufferedImage originalBufferedImage = ImageIO.read(processedFile);

        int matrixSize = calculateImageSizeForScalingBasedOnDifficulty(body.getDifficulty());

        BufferedImage scaledBufferedImage = Scalr.resize(originalBufferedImage, Scalr.Method.ULTRA_QUALITY, Scalr.Mode.FIT_EXACT,
                matrixSize, Scalr.OP_ANTIALIAS);

        File scaledImageFile = new File(outputPath + "/scaled.png");
        ImageIO.write(scaledBufferedImage, "png", scaledImageFile);


        BufferedImage grayScaleBufferImage = new BufferedImage(scaledBufferedImage.getWidth(), scaledBufferedImage.getHeight(),
                BufferedImage.TYPE_BYTE_GRAY);
        Graphics graphics = grayScaleBufferImage.getGraphics();

        graphics.setColor(java.awt.Color.WHITE);
        graphics.fillRect(0, 0, grayScaleBufferImage.getWidth(), grayScaleBufferImage.getHeight());
        graphics.drawImage(scaledBufferedImage, 0, 0, null);
        graphics.dispose();

        long totalBrightness = calculateTotalBrightnessOfImage(grayScaleBufferImage);
        int pixelCount = grayScaleBufferImage.getWidth() * grayScaleBufferImage.getHeight();

        int threshold = (int) (totalBrightness / pixelCount) + body.getContrast();

        BufferedImage blackAndWhiteBufferedImage = generateBlackAndWhiteImage(grayScaleBufferImage, threshold);

        File blackAndWhiteImageFile = new File(outputPath + "/black-and-white-image.png");
        ImageIO.write(blackAndWhiteBufferedImage, "png", blackAndWhiteImageFile);

        boolean[][] nonogram = generateNonogram(blackAndWhiteBufferedImage);

        // if (originalImageFile.delete()) System.out.println("Deleted original");
        // if (processedFile.delete()) System.out.println("Deleted masked");

        byte[] fileContent = FileUtils.readFileToByteArray(blackAndWhiteImageFile);
        String encodedString = Base64.getEncoder().encodeToString(fileContent);

        return new nonogramResponseDto(nonogram, encodedString);
    }

    private void applySaliencyMask(String inputPath, String outputPath) {
        if (this.net == null || this.net.empty()) {
            System.err.println("skip background removal cause model not loaded");
            try {
                FileUtils.copyFile(new File(inputPath), new File(outputPath));
            } catch (IOException e) { e.printStackTrace(); }
            return;
        }

        Mat image = Imgcodecs.imread(inputPath);
        if (image.empty()) return;

        Mat blob = Dnn.blobFromImage(image, 0.011, new Size(320, 320), new Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        Mat output = net.forward();

        Mat score = output.reshape(1, 320);

        Mat mask = new Mat();
        Imgproc.resize(score, mask, image.size());

        Mat binaryMask = new Mat();
        Imgproc.threshold(mask, binaryMask, 0.5, 1, Imgproc.THRESH_BINARY);

        binaryMask.convertTo(binaryMask, CvType.CV_8U, 255);

        Imgcodecs.imwrite(outputPath, binaryMask);
    }

    private int calculateImageSizeForScalingBasedOnDifficulty(Difficulty difficulty){
        return 20 + (10 * difficulty.ordinal());
    }

    private int calculatePixelBrightness(BufferedImage source, int imageYIndex, int imageXIndex) {
        int rgb = source.getRGB(imageXIndex, imageYIndex);
        Color pixelColorRGB = new Color(rgb);

        return (pixelColorRGB.getRed() + pixelColorRGB.getGreen() + pixelColorRGB.getBlue()) / 3;
    }

    private int calculateTotalBrightnessOfImage(BufferedImage grayScaleBufferImage){
        long totalBrightness = 0;
        for (int imageYIndex = 0; imageYIndex < grayScaleBufferImage.getHeight(); imageYIndex++) {
            for (int imageXIndex = 0; imageXIndex < grayScaleBufferImage.getWidth(); imageXIndex++) {
                totalBrightness += calculatePixelBrightness(grayScaleBufferImage, imageYIndex, imageXIndex);
            }
        }

        return (int)totalBrightness;
    }

    private BufferedImage generateBlackAndWhiteImage(BufferedImage grayScaleBufferImage, int threshold){
        BufferedImage blackAndWhiteImage = new BufferedImage(
                grayScaleBufferImage.getWidth(),
                grayScaleBufferImage.getHeight(),
                BufferedImage.TYPE_BYTE_BINARY
        );

        int whiteColor = 0xFFFFFF;
        int blackColor = 0x000000;

        for (int imageYIndex = 0; imageYIndex < grayScaleBufferImage.getHeight(); imageYIndex++) {
            for (int imageXIndex = 0; imageXIndex < grayScaleBufferImage.getWidth(); imageXIndex++) {
                int brightness = calculatePixelBrightness(grayScaleBufferImage, imageYIndex, imageXIndex);
                int newPixel = (brightness >= threshold) ? whiteColor : blackColor;
                blackAndWhiteImage.setRGB(imageXIndex, imageYIndex, newPixel);
            }
        }

        return blackAndWhiteImage;
    }

    private boolean[][] generateNonogram(BufferedImage blackAndWhiteImage){
        boolean[][] nonogram = new boolean[blackAndWhiteImage.getWidth()][blackAndWhiteImage.getHeight()];

        for (int imageXIndex = 0; imageXIndex < blackAndWhiteImage.getWidth(); imageXIndex++) {
            for (int imageYIndex = 0; imageYIndex < blackAndWhiteImage.getHeight(); imageYIndex++) {
                int rgb = blackAndWhiteImage.getRGB(imageXIndex, imageYIndex);

                Color color = new Color(rgb);
                boolean isBlack = (color.getRed() < 128);

                nonogram[imageXIndex][imageYIndex] = isBlack;
            }
        }

        return nonogram;
    }
}