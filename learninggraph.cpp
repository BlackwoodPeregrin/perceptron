#include "learninggraph.h"

#include "ui_learninggraph.h"

LearningGraph::LearningGraph(QWidget *parent)
    : QDialog(parent), ui(new Ui::LearningGraph), scene_(new QGraphicsScene()) {
  ui->setupUi(this);
  ui->graphicsView->setScene(scene_);
}

LearningGraph::~LearningGraph() {
  delete ui;
  delete scene_;
  for (auto i : sub_label) delete i;
}

void LearningGraph::Plot() {
  for (auto i : sub_label) delete i;
  sub_label.clear();
  scene_->clear();
  for (unsigned i{}; i < values_.size(); i++) {
    double width{320}, height{210};
    scene_->addLine(
        width * i / (double)(values_.size() + 1), height * values_[i] * 0.8,
        width * i / (double)(values_.size() + 1), height * 0.8 + 0.01,
        QPen(Qt::black, 10, Qt::SolidLine, Qt::RoundCap));
    sub_label.push_back(
        scene_->addText(QString::number(1 - values_[i], 'f', 2)));
    sub_label[i]->setPos(width * i / (double)(values_.size() + 1) - 15,
                         height * 0.85);
  }
}
