#include "mainwindow.h"

#include <QFileDialog>
#include <QMessageBox>
#include <chrono>

#include "ui_dialogQuestion.h"
#include "ui_mainwindow.h"

namespace s21_network {

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      ui(new Ui::MainWindow),
      timer_(new QTimer),
      scene_(new PaintScene),
      controller_(new Controller) {
  ui->setupUi(this);
  BuildGroupButton();  //  добавляем все кнопки в общую группу
  ui->graphicsView->setScene(scene_);
  SetSlots();  // связываем сигналы со слотами
  timer_->start(100);
}

MainWindow::~MainWindow() {
  delete ui;
  delete timer_;
  delete scene_;
  delete controller_;
}

void MainWindow::KeyPressEvent(QKeyEvent *key) {
  if (key->key() == Qt::Key_Escape) {
    close();
  }
}

void MainWindow::ButChooseTestFileClick() {
  QString filename = QFileDialog::getOpenFileName(
      this, "Select a file to open...", QDir::homePath(), "*.csv");
  if (!filename.isEmpty()) {
    name_test_file_ = filename;
    QStringList nameTestFile = filename.split("/");
    ui->label_name_test_file->setText(nameTestFile.back());
  }
}

void MainWindow::ButChooseTrainFileClick() {
  QString filename = QFileDialog::getOpenFileName(
      this, "Select a file to open...", QDir::homePath(), "*.csv");
  if (!filename.isEmpty()) {
    name_train_file_ = filename;
    QStringList nameTrainFile = filename.split("/");
    ui->label_name_train_file->setText(nameTrainFile.back());
  }
}

void MainWindow::ButLoadWeightsClick() {
  QString filename = QFileDialog::getOpenFileName(
      this, "Select a file to open...", QDir::homePath(), "*.txt");
  if (!filename.isEmpty()) {
    int indexNetwork = ui->list_hidden_layers->currentIndex();
    std::string reultsLoad =
        controller_->LoadWeightsNetwork(filename.toStdString(), indexNetwork);
    QMessageBox::information(this, "Info about loaded weights!",
                             QString::fromLocal8Bit(reultsLoad.c_str()));
    if (reultsLoad == "Weights load SUCSESS") {
      QPixmap pixmap(QPixmap(QString::fromUtf8(":/icons_iu/+.png")));
      switch (indexNetwork) {
        case 0:  // two hiiden layers
          ui->label_name_twoLayers_file->setPixmap(pixmap);
          break;
        case 1:  // three hiiden layers
          ui->label_name_threeLayers_file->setPixmap(pixmap);
          break;
        case 2:  // four hiiden layers
          ui->label_name_fourLayers_file->setPixmap(pixmap);
          break;
        case 3:  // five hiiden layers
          ui->label_name_fiveLayers_file->setPixmap(pixmap);
          break;
      }
    }
  }
}

void MainWindow::ButSaveWeightsClick() {
  QString filename = QFileDialog::getSaveFileName(
      this, "Select a file to open...", QDir::homePath(), "*.txt");
  if (!filename.isEmpty()) {
    controller_->SaveWeightsNetwork(filename.toStdString());
  }
}

void MainWindow::ButLoadBmpClick() {
  QString filename = QFileDialog::getOpenFileName(
      this, "Select a file to open...", QDir::homePath(), "*.bmp");
  if (!filename.isEmpty()) {
    QPixmap pixmap(filename),
        pxm_scaled(pixmap.scaled(
            ui->graphicsView->width() - 20, ui->graphicsView->height() - 20,
            Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
    scene_->clear();
    scene_->addPixmap(pxm_scaled);
    QImage imgBmp = pixmap.toImage().scaled(28, 28, Qt::IgnoreAspectRatio,
                                            Qt::SmoothTransformation);

    std::vector<unsigned> inputLayer;
    for (int i = 0; i < 28; ++i) {
      for (int j = 0; j < 28; ++j) {
        inputLayer.push_back(imgBmp.pixelColor(i, j).black());
      }
    }
    PrintResultNetwork(controller_->get_result_network(inputLayer));
  }
}

void MainWindow::ButStartTestClick() {
  if (name_test_file_.isEmpty()) {
    QMessageBox::information(this, "Info about loaded weights!",
                             "Test file don't Choose, Please choose the test "
                             "file before start test");
  } else {
    double sample_percent = ui->sample_precent->value();
    auto start = std::chrono::high_resolution_clock::now();
    S21Matrix conf_mx = controller_->StartConfusionTest(
        name_test_file_.toStdString(), sample_percent);
    auto dur = std::chrono::high_resolution_clock::now() - start;
    auto mseconds =
        (std::chrono::duration_cast<std::chrono::milliseconds>(dur)).count();

    ui->label_accuracy->setText(
        QString::number(controller_->CalcAccuracy(conf_mx), 'g', 2));
    double prec = controller_->CalcPrecision(conf_mx);
    double recall = controller_->CalcRecall(conf_mx);
    ui->label_precision_value->setText(QString::number(prec, 'g', 2));
    ui->label_recall_value->setText(QString::number(recall, 'g', 2));
    ui->label_fmeasure_value->setText(
        QString::number(controller_->CalcFMeasure(prec, recall), 'g', 2));
    ui->label_time_value->setText(QString::number(mseconds / 1000.0) + " s");
  }
}

void MainWindow::ButStartLearnClick() {
  if (name_train_file_.isEmpty()) {
    QMessageBox::information(this, "Info about loaded weights!",
                             "Train file don't Choose, Please choose the train "
                             "file before learning Network");
  } else {
    dialog_.show();
  }
}

void MainWindow::PrintResultNetwork(const size_t &ch) {
  QString alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  QString letter{};
  letter += alphabet[(unsigned)ch - 1];
  /*---минус один потому что значнеия нейронов начинаются с единицы---*/
  ui->label_char->setText(letter);
}

void MainWindow::SetSlots() {
  connect(scene_, SIGNAL(CharDrawed()), this, SLOT(SlotSymbolDrawed()));
  connect(scene_, SIGNAL(CharClear()), this, SLOT(SlotSymbolClear()));
  connect(timer_, &QTimer::timeout, this, &MainWindow::SlotTimer);
  connect(&group_button_, SIGNAL(idClicked(int)), this,
          SLOT(SlotPushButtonClick(int)));
  connect(ui->radio_but_matrixNet, SIGNAL(clicked()), this,
          SLOT(SlotButtonMatrixNetClicked()));
  connect(ui->radio_but_graphNet, SIGNAL(clicked()), this,
          SLOT(SlotButtonGraphNetClicked()));
  connect(ui->list_hidden_layers, SIGNAL(currentIndexChanged(int)), this,
          SLOT(SlotComboBoxLayersChange(int)));
  connect(&dialog_, SIGNAL(ButtonClick(int)), this,
          SLOT(SlotLearningNetwork(int)));
}

void MainWindow::BuildGroupButton() {
  group_button_.addButton(ui->but_choose_test_file, idButton::TestFile);
  group_button_.addButton(ui->but_choose_train_file, idButton::TrainFile);
  group_button_.addButton(ui->but_load_weights, idButton::LoadWeights);
  group_button_.addButton(ui->but_save_weights, idButton::SaveWeights);
  group_button_.addButton(ui->but_load_bmp, idButton::BmpFile);
  group_button_.addButton(ui->but_start_test, idButton::StartTest);
  group_button_.addButton(ui->but_start_learning, idButton::StartLern);
}

void MainWindow::SlotTimer() {
  timer_->stop();
  scene_->setSceneRect(0, 0, ui->graphicsView->width() - 20,
                       ui->graphicsView->height() - 20);
}

void MainWindow::SlotSymbolDrawed() {
  QPixmap pixels = ui->graphicsView->grab();
  QImage image = pixels.toImage().scaled(28, 28, Qt::IgnoreAspectRatio,
                                         Qt::SmoothTransformation);

  std::vector<unsigned> inputLayer;
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      inputLayer.push_back(image.pixelColor(i, j).black());
    }
  }
  PrintResultNetwork(controller_->get_result_network(inputLayer));
}

void MainWindow::SlotSymbolClear() { ui->label_char->setText(""); }

void MainWindow::SlotPushButtonClick(const int &id_button) {
  if (id_button == idButton::TestFile) {
    ButChooseTestFileClick();
  } else if (id_button == idButton::TrainFile) {
    ButChooseTrainFileClick();
  } else if (id_button == idButton::LoadWeights) {
    ButLoadWeightsClick();
  } else if (id_button == idButton::SaveWeights) {
    ButSaveWeightsClick();
  } else if (id_button == idButton::BmpFile) {
    ButLoadBmpClick();
  } else if (id_button == idButton::StartTest) {
    ButStartTestClick();
  } else if (id_button == idButton::StartLern) {
    ButStartLearnClick();
  }
}

void MainWindow::SlotComboBoxLayersChange(const int &current_index) {
  int hidden_layers = current_index;
  bool type_network;
  if (ui->radio_but_matrixNet->isChecked()) {
    type_network = typeNetwork::Matrix;
  } else if (ui->radio_but_graphNet->isChecked()) {
    type_network = typeNetwork::Graph;
  }
  controller_->SwitchNetwork(hidden_layers, type_network);
}

void MainWindow::SlotButtonMatrixNetClicked() {
  int hidden_layers = ui->list_hidden_layers->currentIndex();
  controller_->SwitchNetwork(hidden_layers, typeNetwork::Matrix);
}

void MainWindow::SlotButtonGraphNetClicked() {
  int hidden_layers = ui->list_hidden_layers->currentIndex();
  controller_->SwitchNetwork(hidden_layers, typeNetwork::Graph);
}

void MainWindow::SlotLearningNetwork(const int &button) {
  int sum_epoch =
      ui->list_epochs->currentIndex() + 1;  // так как индексы начинаются с нуля
  bool continue_learn;
  if (button == DialogQuestion::button::StartOver) {
    continue_learn = false;
  } else if (button == DialogQuestion::button::Continie) {
    continue_learn = true;
  }
  std::vector<double> graph_values{};
  if (ui->check_cross_validation->isChecked()) {
    //  crossvalidation scenario here
    graph_values = controller_->StartCVLearn(
        name_train_file_.toStdString(),
        (ui->list_cross_validation->currentIndex() + 1) * 5, continue_learn);
  } else {
    graph_values = controller_->StartLearnNetwork(
        name_train_file_.toStdString(), sum_epoch, continue_learn,
        name_test_file_.toStdString());
  }
  if (graph_values.size() > 1) {
    learning_graph_.set_values(graph_values);
    learning_graph_.Plot();
    learning_graph_.show();
  }
  QPixmap pixmap(QPixmap(QString::fromUtf8(":/icons_iu/-.png")));
  if (ui->list_hidden_layers->currentText() == "2") {
    ui->label_name_twoLayers_file->setPixmap(pixmap);
  } else if (ui->list_hidden_layers->currentText() == "3") {
    ui->label_name_threeLayers_file->setPixmap(pixmap);
  } else if (ui->list_hidden_layers->currentText() == "4") {
    ui->label_name_fourLayers_file->setPixmap(pixmap);
  } else if (ui->list_hidden_layers->currentText() == "5") {
    ui->label_name_fiveLayers_file->setPixmap(pixmap);
  }
}

DialogQuestion::DialogQuestion(QWidget *parent)
    : QDialog(parent), ui(new Ui::Dialog) {
  ui->setupUi(this);
  group.addButton(ui->but_start_over, button::StartOver);
  group.addButton(ui->but_continue, button::Continie);
  connect(&group, SIGNAL(idClicked(int)), this, SLOT(SlotPushButtonClick(int)));
}

DialogQuestion::~DialogQuestion() { delete ui; }

void DialogQuestion::SlotPushButtonClick(const int &id_button) {
  if (id_button == button::StartOver) {
    emit ButtonClick(button::StartOver);
  } else if (id_button == button::Continie) {
    emit ButtonClick(button::Continie);
  }
  hide();
}

}  // namespace s21_network
