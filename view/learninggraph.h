#pragma once

#include <QDialog>
#include <QGraphicsScene>
#include <QGraphicsTextItem>

namespace Ui {
class LearningGraph;
}

class LearningGraph : public QDialog {
  Q_OBJECT

 public:
  explicit LearningGraph(QWidget *parent = nullptr);
  ~LearningGraph();

  void set_values(const std::vector<double> &src) { values_ = src; }

  void Plot();

 private:
  Ui::LearningGraph *ui;
  std::vector<double> values_;
  std::vector<QGraphicsTextItem *> sub_label;
  QGraphicsScene *scene_;
};
