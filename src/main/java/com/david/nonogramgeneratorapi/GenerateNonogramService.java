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
            System.load("/Users/david/Documents/nonogramGeneratorAPI/src/main/resources/libopencv_java4120.dylib"); // TODO change the dir so it will work in container also
        } catch (UnsatisfiedLinkError e) {
            throw new UnsatisfiedLinkError("Can't load openCV jar files. Error message: " + e);
        }
    }

    @PostConstruct
    public void initModel() throws Exception {
        try {
            this.net = Dnn.readNetFromONNX(modelPath);
            if (this.net.empty()) {
                throw new Exception("Could not load model. Model variable empty after trying to load it.");
            }
        } catch (Exception e) {
            throw new Exception("Exception loading model: " + e);
        }
    }

    public nonogramResponseDto generateNonogram(nonogramGenerationRequestDto requestBody) throws Exception {
        byte[] decodedBytes = Base64.getDecoder().decode(requestBody.getImageBase64());
        String outputPath = "src/main/resources/";

        File originalImageFile = new File(outputPath + "original-image.jpg");
        FileUtils.writeByteArrayToFile(originalImageFile, decodedBytes);

        BufferedImage originalBufferImage = ImageIO.read(originalImageFile);

        String maskedImagePath = outputPath + "masked-image.png";
        detectMainObjectUsingModel(originalImageFile.getAbsolutePath(), maskedImagePath);

        File processedFile = new File(maskedImagePath);
        BufferedImage maskFromModelBufferedImage = ImageIO.read(processedFile);

        BufferedImage modifiedBufferedImage = applyMaskFromModel(originalImageFile.getAbsoluteFile(), maskFromModelBufferedImage);

        int matrixSize = requestBody.getDifficulty().matrixSize;

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

        int threshold = calculateAverageBrightnessOfImage(originalScaled, matrixSize);

        BufferedImage blackAndWhiteBufferedImage = generateBlackAndWhiteImage(grayScaleBufferImage, threshold);

        File blackAndWhiteImageFile = new File(outputPath + "/black-and-white-image.png");
        ImageIO.write(blackAndWhiteBufferedImage, "png", blackAndWhiteImageFile);

        boolean[][] nonogram = generateNonogram(blackAndWhiteBufferedImage);

        if (!originalImageFile.delete()) throw new Exception("Could not delete " + originalImageFile + " file");
        if (!processedFile.delete()) throw new Exception("Could not delete " + processedFile + " file");

        byte[] fileContent = FileUtils.readFileToByteArray(blackAndWhiteImageFile);
        String encodedString = Base64.getEncoder().encodeToString(fileContent);

        if (!blackAndWhiteImageFile.delete()) throw new Exception("Could not delete " + blackAndWhiteImageFile + " file");

        return new nonogramResponseDto(nonogram, encodedString);
    }

    private void detectMainObjectUsingModel(String inputPath, String outputPath) throws Exception {
        if (this.net == null || this.net.empty()) {
            System.err.println("skip background removal cause model not loaded");
            try {
                FileUtils.copyFile(new File(inputPath), new File(outputPath));
            } catch (IOException e) {
                throw new IOException(e.getMessage());
            }
            return;
        }

        Mat imageFromInputPath = Imgcodecs.imread(inputPath);
        if (imageFromInputPath.empty()) throw new Exception("Problem while loading original image for model in: 'detectMainObjectUsingModel");

        Mat mainObjectFromModel = Dnn.blobFromImage(imageFromInputPath, 0.01, new Size(250, 250), new Scalar(0, 0, 0), true, false);
        net.setInput(mainObjectFromModel);

        Mat originalMatOfMainObject = net.forward();

        Mat reshapedMatOfMainObject = originalMatOfMainObject.reshape(1, 250);

        Mat resizedMatOfMainObject = new Mat();
        Imgproc.resize(reshapedMatOfMainObject, resizedMatOfMainObject, imageFromInputPath.size());

        Mat binaryMatOfMainObject = new Mat();
        Imgproc.threshold(resizedMatOfMainObject, binaryMatOfMainObject, 0.5, 1, Imgproc.THRESH_BINARY);
        binaryMatOfMainObject.convertTo(binaryMatOfMainObject, CvType.CV_8U, 255);

        Imgcodecs.imwrite(outputPath, binaryMatOfMainObject);
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

        final int originalRed = color.getRed();
        final int originalGreen = color.getGreen();
        final int originalBlue = color.getBlue();
        final int alpha = color.getAlpha();

        int newRed = Math.min(255, Math.max(0, (int) (originalRed * 0.2)));
        int newGreen = Math.min(255, Math.max(0, (int) (originalGreen * 0.2)));
        int newBlue = Math.min(255, Math.max(0, (int) (originalBlue * 0.2)));

        Color dimmedColor = new Color(newRed, newGreen, newBlue, alpha);

        return isNotMainObjectPixel ? originalPixel : dimmedColor.getRGB();
    }

    private int calculatePixelBrightness(BufferedImage source, int imageYIndex, int imageXIndex) {
        int rgb = source.getRGB(imageXIndex, imageYIndex);
        Color pixelColorRGB = new Color(rgb);

        return (pixelColorRGB.getRed() + pixelColorRGB.getGreen() + pixelColorRGB.getBlue()) / 3;
    }

    private int calculateAverageBrightnessOfImage(BufferedImage grayScaleBufferImage, int matrixSize){
        long totalBrightness = 0;
        for (int imageYIndex = 0; imageYIndex < grayScaleBufferImage.getHeight(); imageYIndex++) {
            for (int imageXIndex = 0; imageXIndex < grayScaleBufferImage.getWidth(); imageXIndex++) {
                totalBrightness += calculatePixelBrightness(grayScaleBufferImage, imageYIndex, imageXIndex);
            }
        }

        int pixelCount = matrixSize * matrixSize;

        return (int) (totalBrightness / pixelCount);
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
                boolean isPixelBlack = (color.getRed() < 128);

                nonogram[imageXIndex][imageYIndex] = isPixelBlack;
            }
        }

        return nonogram;
    }
}