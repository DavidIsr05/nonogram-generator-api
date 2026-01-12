package com.david.nonogramgeneratorapi;

import java.io.FileNotFoundException;

public class CouldNotLoadModelException extends FileNotFoundException {
    CouldNotLoadModelException(String message){
        super((message));
    }
}
