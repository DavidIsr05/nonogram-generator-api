package com.david.nonogramgeneratorapi;

import java.io.FileNotFoundException;

public class CouldNotLoadModelException extends FileNotFoundException {
    CouldNotLoadModelException(){
        super(("Could not load model. Model variable empty after trying to load it."));
    }
}
