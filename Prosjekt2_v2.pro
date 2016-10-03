TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -larmadillo -llapack -lblas

SOURCES += main.cpp
