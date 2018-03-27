#QT += widgets

LIBS += `pkg-config --libs opencv`
SOURCES += \
    main.cpp

HEADERS += \
    gms_matcher.h \
    header.h
