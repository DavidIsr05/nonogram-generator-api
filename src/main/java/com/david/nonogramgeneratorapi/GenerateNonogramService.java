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
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.FileSystemException;
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
                throw new CouldNotLoadModelException("Could not load model. Model variable empty after trying to load it.");
            }
        } catch (CouldNotLoadModelException e) {
            throw new CouldNotLoadModelException("Exception loading model: " + e);
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

        BufferedImage modifiedBufferedImage = applyMaskFromModel(originalImageFile.getAbsoluteFile(), maskFromModelBufferedImage, requestBody.getPixelHighlightValue());

        int matrixSize = requestBody.getDifficulty().getMatrixSize();

        BufferedImage scaledBufferedImage = Scalr.resize(modifiedBufferedImage, Scalr.Method.ULTRA_QUALITY, Scalr.Mode.FIT_EXACT,
                matrixSize, Scalr.OP_ANTIALIAS);

        BufferedImage originalScaled = Scalr.resize(originalBufferImage, Scalr.Method.ULTRA_QUALITY, Scalr.Mode.FIT_EXACT,
                matrixSize, Scalr.OP_ANTIALIAS);

        File originalDownscaledFile = new File(outputPath + "/original-downscaled.png");
        ImageIO.write(originalScaled, "png", originalDownscaledFile);

        BufferedImage grayScaleBufferImage = new BufferedImage(matrixSize, matrixSize, BufferedImage.TYPE_BYTE_GRAY);
        Graphics graphics = grayScaleBufferImage.getGraphics();

        graphics.setColor(java.awt.Color.WHITE);
        graphics.fillRect(0, 0, matrixSize, matrixSize);
        graphics.drawImage(scaledBufferedImage, 0, 0, null);
        graphics.dispose();

        int threshold = calculateAverageBrightnessOfImage(originalScaled, matrixSize);

        BufferedImage blackAndWhiteBufferedImage = generateBlackAndWhiteImage(grayScaleBufferImage, threshold);

        boolean[][] nonogram = generateNonogram(blackAndWhiteBufferedImage);

        BufferedImage originalImageDownscaledForPreview = originalBufferImage;

        if (originalBufferImage.getHeight() > 500 | originalBufferImage.getWidth() > 500) {
            originalImageDownscaledForPreview = Scalr.resize(originalBufferImage, Scalr.Method.ULTRA_QUALITY, Scalr.Mode.AUTOMATIC,
                    500, Scalr.OP_ANTIALIAS);
        }
        File previewFile = new File(outputPath + "/previw.png");
        ImageIO.write(highlightOriginalImageBasedOnBlackAndWhiteImage(blackAndWhiteBufferedImage, originalImageDownscaledForPreview, threshold), "png", previewFile);

        if (!originalImageFile.delete()) throw new CouldNotDeleteFileException("Could not delete " + originalImageFile + " file with path: " + originalImageFile.getAbsolutePath());
        if (!processedFile.delete()) throw new CouldNotDeleteFileException("Could not delete " + processedFile + " file with path: " + processedFile.getAbsolutePath());

        byte[] previewFileContent = FileUtils.readFileToByteArray(previewFile);
        String previewImageBase64 = Base64.getEncoder().encodeToString(previewFileContent);

        byte[] originalDownscaledFileContent = FileUtils.readFileToByteArray(originalDownscaledFile);
        String originalDownscaledImageBase64 = Base64.getEncoder().encodeToString(originalDownscaledFileContent);

        if (!previewFile.delete()) throw new CouldNotDeleteFileException("Could not delete " + previewFile + " file with path: " + previewFile.getAbsolutePath());
        if (!originalDownscaledFile.delete()) throw new CouldNotDeleteFileException("Could not delete " + originalDownscaledFile + " file with path: " + originalDownscaledFile.getAbsolutePath());

        return new nonogramResponseDto(nonogram, previewImageBase64, originalDownscaledImageBase64);
    }

    private void detectMainObjectUsingModel(String inputPath, String outputPath) throws Exception {
        if (this.net == null || this.net.empty()) {
            System.err.println("skip background removal cause model not loaded");
            try {
                FileUtils.copyFile(new File(inputPath), new File(outputPath));
            } catch (FileSystemException e) {
                throw new FileSystemException(e.getMessage());
            }
            return;
        }

        Mat imageFromInputPath = Imgcodecs.imread(inputPath);
        if (imageFromInputPath.empty()) throw new FileNotFoundException("Problem while loading original image for model in: 'detectMainObjectUsingModel', with path: " + inputPath);

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

    private BufferedImage applyMaskFromModel(File originalImageFile, BufferedImage maskFromModelBufferedImage, double pixelHighlightValue) throws IOException {

        BufferedImage originalBufferedImage = ImageIO.read(originalImageFile);

        for (int imageYIndex = 0; imageYIndex < originalBufferedImage.getHeight(); imageYIndex++) {
            for (int imageXIndex = 0; imageXIndex < originalBufferedImage.getWidth(); imageXIndex++) {
                boolean isNotMainObjectPixel = calculatePixelBrightness(maskFromModelBufferedImage, imageYIndex, imageXIndex) == 0;
                int originalPixel = originalBufferedImage.getRGB(imageXIndex, imageYIndex);

                int updatedPixel = getUpdatedPixel(originalPixel, isNotMainObjectPixel, pixelHighlightValue);

                originalBufferedImage.setRGB(imageXIndex, imageYIndex, updatedPixel);
            }
        }

        return originalBufferedImage;
    }

    private static int getUpdatedPixel(int originalPixel, boolean isNotMainObjectPixel, double pixelHighlightValue) {
        if (!isNotMainObjectPixel) {
            Color color = new Color(originalPixel, true);

            final int newRed = Math.min(255, Math.max(0, (int) (color.getRed() * pixelHighlightValue)));
            final int newGreen = Math.min(255, Math.max(0, (int) (color.getGreen() * pixelHighlightValue)));
            final int newBlue = Math.min(255, Math.max(0, (int) (color.getBlue() * pixelHighlightValue)));
            final int alpha = color.getAlpha();

            Color dimmedColor = new Color(newRed, newGreen, newBlue, alpha);
            originalPixel = dimmedColor.getRGB();
        }

        return originalPixel;
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

    private BufferedImage highlightOriginalImageBasedOnBlackAndWhiteImage(BufferedImage blackAndWhiteImage, BufferedImage originalImage, int threshold){
        int pixelWidthRatio = originalImage.getWidth() / blackAndWhiteImage.getWidth();
        int pixelHeightRatio = originalImage.getHeight() / blackAndWhiteImage.getHeight();

        BufferedImage previewOverlayBufferImage = new BufferedImage(originalImage.getWidth(), originalImage.getHeight(),
                BufferedImage.TYPE_INT_ARGB);

        Graphics2D g2d = previewOverlayBufferImage.createGraphics();
        g2d.setComposite(AlphaComposite.Clear);
        g2d.fillRect(0, 0, previewOverlayBufferImage.getWidth(), previewOverlayBufferImage.getHeight());


        for (int blackAndWhiteImageXIndex = 0; blackAndWhiteImageXIndex < blackAndWhiteImage.getWidth(); blackAndWhiteImageXIndex++){
            for (int blackAndWhiteImageYIndex = 0; blackAndWhiteImageYIndex < blackAndWhiteImage.getHeight(); blackAndWhiteImageYIndex++){

                boolean isPixelBlack = calculatePixelBrightness(blackAndWhiteImage, blackAndWhiteImageYIndex, blackAndWhiteImageXIndex) < 128;

                if (isPixelBlack){

                    int coordinateXOnOriginalBasedOnBlackAndWhiteImage = blackAndWhiteImageXIndex*pixelWidthRatio;
                    int coordinateYOnOriginalBasedOnBlackAndWhiteImage = blackAndWhiteImageYIndex*pixelHeightRatio;

                    float pixelHighlightValueBasedOnImageAverageBrightness = threshold < 128 ? 0.2f : 0.6f;

                    g2d.setColor(Color.RED);
                    g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, pixelHighlightValueBasedOnImageAverageBrightness));
                    g2d.drawOval(coordinateXOnOriginalBasedOnBlackAndWhiteImage, coordinateYOnOriginalBasedOnBlackAndWhiteImage, pixelWidthRatio, pixelHeightRatio);
                }
            }
        }
        g2d.dispose();

        return previewOverlayBufferImage;
    }
}