// C++ 面向对象进阶示例：抽象基类 / 继承 / 覆盖 / 多态
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// 抽象基类：接口 + 共享行为
class Shape
{
public:
  // 虚析构，保证通过基类指针删除派生类对象时行为正确
  virtual ~Shape() = default; // =default 展示“规则零/五”中的默认特性

  // 纯虚函数：接口
  virtual std::string name() const = 0;
  virtual double area() const = 0;
  virtual void draw() const = 0;

  // 非虚函数里调用虚函数：NVI 惯用法的简单展示
  std::string info() const
  { // const 成员函数不修改对象状态
    return name() + ", area=" + std::to_string(area());
  }

  // 为每个对象分配一个只读 id，演示受保护成员与静态成员
  int id() const { return id_; }

protected:
  explicit Shape(int id) : id_(id) {}

  static int next_id_; // 静态成员：所有实例共享
  int id_{};           // 受保护：派生类可见，外部不可见
};

int Shape::next_id_ = 1; // 静态成员定义与初始化

// 圆形：实现抽象接口
class Circle : public Shape
{
public:
  explicit Circle(double r)
      : Shape(next_id_++), radius_(r)
  {
    if (r <= 0)
      throw std::invalid_argument("radius must be > 0");
  }

  std::string name() const override { return "Circle#" + std::to_string(id()); }
  double area() const override { return 3.14159265358979323846 * radius_ * radius_; }
  void draw() const override { std::cout << "(circle r=" << radius_ << ")\n"; }

  double radius() const { return radius_; } // 只读访问器

private:
  double radius_{}; // 封装：私有数据
};

// 矩形
class Rectangle : public Shape
{
public:
  Rectangle(double w, double h)
      : Shape(next_id_++), width_(w), height_(h)
  {
    if (w <= 0 || h <= 0)
      throw std::invalid_argument("width/height must be > 0");
  }

  std::string name() const override { return "Rectangle#" + std::to_string(id()); }
  double area() const override { return width_ * height_; }
  void draw() const override
  {
    std::cout << "+--" << std::string(static_cast<int>(width_), '-') << "--+\n";
    std::cout << "|  (" << width_ << "x" << height_ << ")  |\n";
    std::cout << "+--" << std::string(static_cast<int>(width_), '-') << "--+\n";
  }

  double width() const { return width_; }
  double height() const { return height_; }

protected:
  double width_{};
  double height_{};
};

// 正方形：从矩形派生，演示“重用已有实现”，并标记为 final 不允许再被继承
class Square final : public Rectangle
{
public:
  explicit Square(double side)
      : Rectangle(side, side), side_(side) {}

  // 覆盖 name，仅改变标识，其他行为复用 Rectangle
  std::string name() const override { return "Square#" + std::to_string(id()); }
  void draw() const override { std::cout << "[square side=" << side_ << "]\n"; }

  double side() const { return side_; }

private:
  double side_{};
};

// 演示：通过基类引用使用多态
void print_area(const Shape &s)
{
  std::cout << s.name() << " has area = " << s.area() << "\n";
}

// 演示：运算符重载配合非虚接口 info()
std::ostream &operator<<(std::ostream &os, const Shape &s)
{
  return os << s.info();
}

int main()
{
  using std::make_unique;
  std::vector<std::unique_ptr<Shape>> shapes;
  shapes.emplace_back(make_unique<Circle>(2.0));
  shapes.emplace_back(make_unique<Rectangle>(3.0, 4.0));
  shapes.emplace_back(make_unique<Square>(5.0));

  std::cout << "-- Polymorphic dispatch via base pointer --\n";
  for (const auto &p : shapes)
  {
    // 虚函数调用：在运行时根据实际对象类型进行分派
    p->draw();
    print_area(*p);
    std::cout << *p << "\n"; // 使用非虚 info()，其内部仍触发虚调用

    // 演示：安全向下转型（仅当确有需要时）
    if (auto sq = dynamic_cast<const Square *>(p.get()))
    {
      std::cout << "  -> It's a Square, side = " << sq->side() << "\n";
    }
  }

  std::cout << "\n-- Inheritance chain check --\n";
  Shape &s1 = *shapes[0]; // Circle 作为 Shape 引用
  Shape &s2 = *shapes[1]; // Rectangle 作为 Shape 引用
  Shape &s3 = *shapes[2]; // Square 作为 Shape 引用
  std::cout << s1.name() << ", " << s2.name() << ", " << s3.name() << "\n";

  // 通过基类指针删除派生类对象由 unique_ptr 自动完成，
  // 因为基类析构为 virtual，派生对象能被正确析构。

  return 0;
}
