#include "view/mainwindow.h"

#include <QApplication>

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    s21_network::MainWindow w;
    w.show();
    return a.exec();
}
