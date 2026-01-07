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
    String modelPath = "src/main/resources/u2net.onnx";

    static {
        try {
            System.load("/Users/david/Documents/nonogramGeneratorAPI/src/main/resources/libopencv_java4120.dylib");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("could not load openCV jar files: " + e.getMessage());
            System.exit(1);
        }
    }

    @PostConstruct
    public void initModel() {
        try {
            this.net = Dnn.readNetFromONNX(modelPath);
            if (this.net.empty()) {
                System.err.println("failed to load U2Net model");
            }
        } catch (Exception e) {
            System.err.println("Exception loading model: " + e.getMessage());
        }
    }

    public nonogramResponseDto generateNonogram(nonogramGenerationRequestDto body) throws IOException {
        byte[] decodedBytes = Base64.getDecoder().decode(body.getImageBase64());
        String outputPath = "src/main/resources/";

        File originalImageFile = new File(outputPath + "original-image.jpg");
        FileUtils.writeByteArrayToFile(originalImageFile, decodedBytes);

        BufferedImage originalBufferImage = ImageIO.read(originalImageFile);

        String maskedImagePath = outputPath + "masked-image.png";
        applySaliencyMask(originalImageFile.getAbsolutePath(), maskedImagePath);

        File processedFile = new File(maskedImagePath);
        BufferedImage maskFromModelBufferedImage = ImageIO.read(processedFile);

        BufferedImage modifiedBufferedImage = applyMaskFromModel(originalImageFile.getAbsoluteFile(), maskFromModelBufferedImage);

        int matrixSize = calculateImageSizeForScalingBasedOnDifficulty(body.getDifficulty());

        BufferedImage scaledBufferedImage = Scalr.resize(modifiedBufferedImage, Scalr.Method.ULTRA_QUALITY, Scalr.Mode.FIT_EXACT,
                matrixSize, Scalr.OP_ANTIALIAS);

        BufferedImage originalScaled = Scalr.resize(originalBufferImage, Scalr.Method.ULTRA_QUALITY, Scalr.Mode.FIT_EXACT,
                matrixSize, Scalr.OP_ANTIALIAS);

        BufferedImage grayScaleBufferImage = new BufferedImage(matrixSize, matrixSize, BufferedImage.TYPE_BYTE_GRAY);
        Graphics graphics = grayScaleBufferImage.getGraphics();

        graphics.setColor(java.awt.Color.WHITE);
        graphics.fillRect(0, 0, matrixSize, matrixSize);
        graphics.drawImage(scaledBufferedImage, 0, 0, null);
        graphics.dispose();

        long totalBrightness = calculateTotalBrightnessOfImage(originalScaled);
        int pixelCount = matrixSize * matrixSize;

        int threshold = (int) (totalBrightness / pixelCount);
        System.out.println(threshold);

        BufferedImage blackAndWhiteBufferedImage = generateBlackAndWhiteImage(grayScaleBufferImage, threshold);

        File blackAndWhiteImageFile = new File(outputPath + "/black-and-white-image.png");
        ImageIO.write(blackAndWhiteBufferedImage, "png", blackAndWhiteImageFile);

        boolean[][] nonogram = generateNonogram(blackAndWhiteBufferedImage);

        if (originalImageFile.delete()) System.out.println("Deleted original");
        if (processedFile.delete()) System.out.println("Deleted masked");

        byte[] fileContent = FileUtils.readFileToByteArray(blackAndWhiteImageFile);
        String encodedString = Base64.getEncoder().encodeToString(fileContent);

        if (blackAndWhiteImageFile.delete()) System.out.println("Deleted black-and-white");

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

        Mat blob = Dnn.blobFromImage(image, 0.01, new Size(250, 250), new Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        Mat output = net.forward();

        Mat score = output.reshape(1, 250);
        Mat mask = new Mat();
        Imgproc.resize(score, mask, image.size());
        Mat binaryMask = new Mat();
        Imgproc.threshold(mask, binaryMask, 0.5, 1, Imgproc.THRESH_BINARY);

        binaryMask.convertTo(binaryMask, CvType.CV_8U, 255);

        Imgcodecs.imwrite(outputPath, binaryMask);
    }

    private BufferedImage applyMaskFromModel(File originalImageFile, BufferedImage maskFromModelBufferedImage) throws IOException {

        BufferedImage originalBufferedImage = ImageIO.read(originalImageFile);

        for (int imageYIndex = 0; imageYIndex < originalBufferedImage.getHeight(); imageYIndex++) {
            for (int imageXIndex = 0; imageXIndex < originalBufferedImage.getWidth(); imageXIndex++) {
                boolean isNotMainObjectPixel = calculatePixelBrightness(maskFromModelBufferedImage, imageYIndex, imageXIndex) == 0;
                int originalPixel = originalBufferedImage.getRGB(imageXIndex, imageYIndex);

                int updatedPixel = getUpdatedPixel(originalPixel, isNotMainObjectPixel);

                originalBufferedImage.setRGB(imageXIndex, imageYIndex, updatedPixel);
            }
        }

        return originalBufferedImage;
    }

    private static int getUpdatedPixel(int originalPixel, boolean isNotMainObjectPixel) {
        Color color = new Color(originalPixel, true);

        int red = (int) (color.getRed() * 0.2);
        int green = (int) (color.getGreen() * 0.2);
        int blue = (int) (color.getBlue() * 0.2);
        int alpha = color.getAlpha();

        red = Math.min(255, Math.max(0, red));
        green = Math.min(255, Math.max(0, green));
        blue = Math.min(255, Math.max(0, blue));

        Color dimmedColor = new Color(red, green, blue, alpha);

        return isNotMainObjectPixel ? originalPixel : dimmedColor.getRGB();
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