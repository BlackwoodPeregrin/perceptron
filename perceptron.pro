QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    controller/controller.cpp \
    main.cpp \
    model/graphNetwork.cpp \
    model/matrixNetwork.cpp \
    model/network.cpp \
    model/neuron.cpp \
    view/learninggraph.cpp \
    view/mainwindow.cpp

HEADERS += \
    controller/controller.hpp \
    model/graphNetwork.hpp \
    model/interfaceNetwork.hpp \
    model/matrixNetwork.hpp \
    model/network.hpp \
    model/neuron.h \
    model/s21_matrix_oop.h \
    view/learninggraph.h \
    view/mainwindow.h \
    view/paintscene.h

FORMS += \
    view/ui/dialogQuestion.ui \
    view/ui/learninggraph.ui \
    view/ui/mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    icons.qrc

DISTFILES += \
    img/+.png \
    img/-.png
