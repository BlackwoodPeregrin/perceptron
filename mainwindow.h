#pragma once

#include <QButtonGroup>
#include <QDialog>
#include <QKeyEvent>
#include <QMainWindow>

#include "controller.hpp"
#include "learninggraph.h"
#include "paintscene.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
class Dialog;
}  // namespace Ui
QT_END_NAMESPACE

namespace s21_network {
class DialogQuestion : public QDialog {
  Q_OBJECT
 public:
  explicit DialogQuestion(QWidget *parent = nullptr);
  ~DialogQuestion();
  enum button { StartOver, Continie };

 signals:
  void ButtonClick(int);
  void DialogClose();

 private slots:
  void SlotPushButtonClick(const int &id_button);

 private:
  QButtonGroup group;
  Ui::Dialog *ui;
};

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit MainWindow(QWidget *parent = nullptr);
  ~MainWindow();

 private:
  void SetSlots();
  void BuildGroupButton();
  void PrintResultNetwork(const size_t &ch);
  void KeyPressEvent(QKeyEvent *key);

  void ButChooseTestFileClick();
  void ButChooseTrainFileClick();
  void ButLoadWeightsClick();
  void ButSaveWeightsClick();
  void ButLoadBmpClick();
  void ButStartTestClick();
  void ButStartLearnClick();

 private slots:
  void SlotTimer();
  void SlotSymbolDrawed();
  void SlotSymbolClear();
  void SlotPushButtonClick(const int &id_button);
  void SlotComboBoxLayersChange(const int &current_index);
  void SlotButtonMatrixNetClicked();
  void SlotButtonGraphNetClicked();
  void SlotLearningNetwork(const int &button);

 private:
  Ui::MainWindow *ui;
  QTimer *timer_;
  PaintScene *scene_;
  Controller *controller_;
  enum idButton {
    TestFile,
    TrainFile,
    LoadWeights,
    SaveWeights,
    BmpFile,
    StartTest,
    StartLern
  };
  QButtonGroup group_button_;
  QString name_test_file_;
  QString name_train_file_;
  DialogQuestion dialog_;
  LearningGraph learning_graph_;
};
}  // namespace s21_network
