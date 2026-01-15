package com.david.nonogramgeneratorapi;

import com.david.nonogramgeneratorapi.dtos.*;
import jakarta.annotation.PostConstruct;
import org.imgscalr.Scalr;
import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.util.Base64;

@Service
public class GenerateNonogramService {

    private Net modelModule;
    final String MODEL_PATH = "src/main/resources/u2net.onnx";

    static {

        File file = new File("src/main/resources/libopencv_java4120.dylib");

        try {
            System.load(file.getAbsolutePath());
        } catch (UnsatisfiedLinkError e) {
            throw new UnsatisfiedLinkError("Can't load openCV jar files. Error message: " + e);
        }
    }

    @PostConstruct
    public void initModel() {
        modelModule = Dnn.readNetFromONNX(MODEL_PATH);
        if (modelModule.empty()) {
            throw new RuntimeException(new CouldNotLoadModelException("Could not load model. Model variable empty after trying to load it."));
        }
    }

    public nonogramResponseDto generateNonogram(nonogramGenerationRequestDto requestBody) throws Exception {
        byte[] originalImageAsBytes = Base64.getDecoder().decode(requestBody.getImageBase64());

        ByteArrayInputStream originalImageAsByteArrayStream = new ByteArrayInputStream(originalImageAsBytes);

        BufferedImage originalImage = ImageIO.read(originalImageAsByteArrayStream);

        BufferedImage mainObjectFromModel = detectMainObjectUsingModel(originalImage);

        BufferedImage originalImageWithDimmedMainObject = applyDimFactorToImageUsingMainObjectFromModel(originalImage, mainObjectFromModel, requestBody.getMainObjectDimFactor());

        int matrixSize = requestBody.getDifficulty().getMatrixSize();

        BufferedImage downScaledImageWithDimmedMainObject = Scalr.resize(originalImageWithDimmedMainObject, Scalr.Method.ULTRA_QUALITY, Scalr.Mode.FIT_EXACT, matrixSize, Scalr.OP_ANTIALIAS);

        BufferedImage downscaledOriginalImage = Scalr.resize(originalImage, Scalr.Method.ULTRA_QUALITY, Scalr.Mode.FIT_EXACT, matrixSize, Scalr.OP_ANTIALIAS);

        BufferedImage grayScaledImage = new BufferedImage(matrixSize, matrixSize, BufferedImage.TYPE_BYTE_GRAY);

        Graphics graphics = grayScaledImage.getGraphics();

        graphics.setColor(java.awt.Color.WHITE);
        graphics.fillRect(0, 0, matrixSize, matrixSize);
        graphics.drawImage(downScaledImageWithDimmedMainObject, 0, 0, null);
        graphics.dispose();

        int threshold = calculateAverageBrightnessOfImage(downscaledOriginalImage, matrixSize);

        boolean[][] nonogram = generateNonogram(grayScaledImage, threshold);

        BufferedImage downscaledOriginalImageForPreview = originalImage;

        if (originalImage.getHeight() > 500 | originalImage.getWidth() > 500) {
            downscaledOriginalImageForPreview = Scalr.resize(originalImage, Scalr.Method.ULTRA_QUALITY, Scalr.Mode.AUTOMATIC, 500, Scalr.OP_ANTIALIAS);
        }

        BufferedImage previewImage = highlightOriginalImageBasedOnBlackAndWhiteImage(nonogram, downscaledOriginalImageForPreview, threshold, requestBody.getPreviewImageHighlightColor());

        String previewImageBase64 = bufferedImageToBase64(previewImage);

        String downscaledOriginalImageForCompletedNonogramsBase64 = bufferedImageToBase64(downscaledOriginalImage);

        return new nonogramResponseDto(nonogram, previewImageBase64, downscaledOriginalImageForCompletedNonogramsBase64);
    }

    private BufferedImage detectMainObjectUsingModel(BufferedImage inputImage) throws Exception {
        Mat inputImageInMatFormat = bufferedImageToMat(inputImage);

        if (inputImageInMatFormat.empty())
            throw new FileNotFoundException("Problem while loading original image for model in: 'detectMainObjectUsingModel'");

        Mat mainObjectFromModel = Dnn.blobFromImage(inputImageInMatFormat, 0.01, new Size(250, 250), new Scalar(0, 0, 0), true, false);
        modelModule.setInput(mainObjectFromModel);

        Mat originalMatOfMainObject = modelModule.forward();

        Mat reshapedMatOfMainObject = originalMatOfMainObject.reshape(1, 250);

        Mat resizedMatOfMainObjectBasedOnOriginalInputImage = new Mat();

        Imgproc.resize(reshapedMatOfMainObject, resizedMatOfMainObjectBasedOnOriginalInputImage, inputImageInMatFormat.size());

        Mat binaryMatOfMainObject = new Mat();

        Imgproc.threshold(resizedMatOfMainObjectBasedOnOriginalInputImage, binaryMatOfMainObject, 0.5, 1, Imgproc.THRESH_BINARY);

        binaryMatOfMainObject.convertTo(binaryMatOfMainObject, CvType.CV_8U, 255);

        return matToBufferedImage(binaryMatOfMainObject);
    }

    private BufferedImage applyDimFactorToImageUsingMainObjectFromModel(BufferedImage originalImage, BufferedImage mainObjectFromModel, double dimFactor) {
        BufferedImage duplicatedOriginalImage = new BufferedImage(originalImage.getWidth(), originalImage.getHeight(), originalImage.getType());
        Graphics g = duplicatedOriginalImage.getGraphics();
        g.drawImage(originalImage, 0, 0, null);
        g.dispose();

        for (int imageXIndex = 0; imageXIndex < mainObjectFromModel.getWidth(); imageXIndex++) {
            for (int imageYIndex = 0; imageYIndex < mainObjectFromModel.getHeight(); imageYIndex++) {
                boolean isMainObjectPixel = calculatePixelBrightness(mainObjectFromModel, imageXIndex, imageYIndex) != 0;
                int originalPixel = duplicatedOriginalImage.getRGB(imageXIndex, imageYIndex);

                int updatedPixel = getUpdatedPixel(originalPixel, isMainObjectPixel, dimFactor);

                duplicatedOriginalImage.setRGB(imageXIndex, imageYIndex, updatedPixel);
            }
        }

        return duplicatedOriginalImage;
    }

    private int getUpdatedPixel(int originalPixel, boolean isMainObjectPixel, double dimFactor) {
        if (isMainObjectPixel) {
            Color color = new Color(originalPixel, true);

            final int newRed = getUpdatedPixelColor(color.getRed(), dimFactor);
            final int newGreen = getUpdatedPixelColor(color.getGreen(), dimFactor);
            final int newBlue = getUpdatedPixelColor(color.getBlue(), dimFactor);
            final int alpha = color.getAlpha();

            Color dimmedColor = new Color(newRed, newGreen, newBlue, alpha);
            originalPixel = dimmedColor.getRGB();
        }

        return originalPixel;
    }

    private int getUpdatedPixelColor(int RGBColor, double dimFactor){
        int maxPixelColorValue = 255;

        return Math.min(maxPixelColorValue, Math.max(0, (int) (RGBColor * dimFactor)));
    }

    private int calculatePixelBrightness(BufferedImage inputImage, int imageXIndex, int imageYIndex) {
        int rgb = inputImage.getRGB(imageXIndex, imageYIndex);
        Color pixelColorRGB = new Color(rgb);

        return (pixelColorRGB.getRed() + pixelColorRGB.getGreen() + pixelColorRGB.getBlue()) / 3;
    }

    private int calculateAverageBrightnessOfImage(BufferedImage grayScaledImage, int matrixSize) {
        long totalBrightness = 0;

        for (int imageXIndex = 0; imageXIndex < grayScaledImage.getHeight(); imageXIndex++) {
            for (int imageYIndex = 0; imageYIndex < grayScaledImage.getWidth(); imageYIndex++) {
                totalBrightness += calculatePixelBrightness(grayScaledImage, imageXIndex, imageYIndex);
            }
        }

        int pixelCount = (int) Math.pow(matrixSize, 2);

        return (int) (totalBrightness / pixelCount);
    }

    private boolean[][] generateNonogram(BufferedImage grayScaledImage, int threshold) {
        boolean[][] nonogram = new boolean[grayScaledImage.getWidth()][grayScaledImage.getHeight()];

        for (int imageXIndex = 0; imageXIndex < grayScaledImage.getWidth(); imageXIndex++) {
            for (int imageYIndex = 0; imageYIndex < grayScaledImage.getHeight(); imageYIndex++) {

                int brightness = calculatePixelBrightness(grayScaledImage, imageXIndex, imageYIndex);
                boolean isPixelBlack = brightness < threshold;

                nonogram[imageXIndex][imageYIndex] = isPixelBlack;
            }
        }

        return nonogram;
    }

    private BufferedImage highlightOriginalImageBasedOnBlackAndWhiteImage(boolean[][] nonogram, BufferedImage originalImage, int threshold, int previewImageHighlightColor) {
        int pixelWidthRatio = originalImage.getWidth() / nonogram.length;
        int pixelHeightRatio = originalImage.getHeight() / nonogram.length;

        int width = originalImage.getWidth();
        int height = originalImage.getHeight();

        Color highlightColor = new Color(previewImageHighlightColor);

        BufferedImage previewImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);

        Graphics2D g2d = previewImage.createGraphics();
        g2d.setComposite(AlphaComposite.Clear);
        g2d.fillRect(0, 0, width, height);

        final float highDimFactor = 0.2f;
        final float lowDimFactor = 0.6f;

        final int blackAndWhitePixelThreshold = 128;

        for (int nonogramXIndex = 0; nonogramXIndex < nonogram.length; nonogramXIndex++) {
            for (int nonogramYIndex = 0; nonogramYIndex < nonogram[0].length; nonogramYIndex++) {

                if (nonogram[nonogramXIndex][nonogramYIndex]) {

                    int coordinateXOnOriginalBasedOnNonogram = nonogramXIndex * pixelWidthRatio;
                    int coordinateYOnOriginalBasedOnNonogram = nonogramYIndex * pixelHeightRatio;

                    float previewImageOpacityBasedOnOriginalImageAverageBrightness = threshold < blackAndWhitePixelThreshold ? highDimFactor : lowDimFactor;

                    g2d.setColor(highlightColor);
                    g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, previewImageOpacityBasedOnOriginalImageAverageBrightness));
                    g2d.drawOval(coordinateXOnOriginalBasedOnNonogram, coordinateYOnOriginalBasedOnNonogram, pixelWidthRatio, pixelHeightRatio);
                }
            }
        }
        g2d.dispose();

        return previewImage;
    }

    public static Mat bufferedImageToMat(BufferedImage inputImage) {
        BufferedImage duplicatedImage = new BufferedImage(inputImage.getWidth(), inputImage.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
        duplicatedImage.getGraphics().drawImage(inputImage, 0, 0, null);

        byte[] duplicatedImageInBytes = ((DataBufferByte) duplicatedImage.getRaster().getDataBuffer()).getData();

        Mat matFormatImage = new Mat(duplicatedImage.getHeight(), duplicatedImage.getWidth(), CvType.CV_8UC3);
        matFormatImage.put(0, 0, duplicatedImageInBytes);

        return matFormatImage;
    }

    public static BufferedImage matToBufferedImage(Mat inputMatImage) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (inputMatImage.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }

        BufferedImage bufferedImageFormatImage = new BufferedImage(inputMatImage.cols(), inputMatImage.rows(), type);

        byte[] inputImageInBytes = ((DataBufferByte) bufferedImageFormatImage.getRaster().getDataBuffer()).getData();

        inputMatImage.get(0, 0, inputImageInBytes);

        return bufferedImageFormatImage;
    }

    public static String bufferedImageToBase64(BufferedImage inputImage) throws IOException {
        ByteArrayOutputStream inputImageInByteArray = new ByteArrayOutputStream();

        ImageIO.write(inputImage, "png", inputImageInByteArray);

        byte[] imageBytes = inputImageInByteArray.toByteArray();

        return Base64.getEncoder().encodeToString(imageBytes);
    }
}