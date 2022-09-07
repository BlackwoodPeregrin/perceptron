#pragma once

#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QTimer>

namespace s21_network {
class PaintScene : public QGraphicsScene {
  Q_OBJECT

 public:
    explicit PaintScene(QObject *parent = nullptr) : QGraphicsScene(parent) {}

 signals:
    void CharDrawed();
    void CharClear();

 private:
    void mousePressEvent(QGraphicsSceneMouseEvent *event) {
        if (event->buttons() == Qt::LeftButton) {
            addEllipse(event->scenePos().x() - 22,
                       event->scenePos().y() - 22,
                       45,
                       45,
                       QPen(Qt::NoPen),
                       QBrush(Qt::black));
            previous_point_ = event->scenePos();
        } else if (event->buttons() == Qt::RightButton) {
            clear();
            emit CharClear();
        }
    }
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) {
        if (event->buttons() == Qt::LeftButton) {
            addLine(previous_point_.x(),
                    previous_point_.y(),
                    event->scenePos().x(),
                    event->scenePos().y(),
                    QPen(Qt::black, 45, Qt::SolidLine, Qt::RoundCap));
            previous_point_ = event->scenePos();
        }
    }
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
        if (event->button() == Qt::LeftButton) {
            emit CharDrawed();
        }
    }

 private:
    QPointF previous_point_;
};
}  // namespace s21_network
