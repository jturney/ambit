//
// Created by Justin Turney on 3/30/15.
//

#ifndef AMBIT_TIMER_H
#define AMBIT_TIMER_H

#include <string>
#include <map>
#include <tuple>
#include <chrono>

namespace ambit {
namespace timer {

void initialize();

void finalize();

void report();

void timer_push(const std::string& name);
void timer_pop();

}
}

#endif //AMBIT_TIMER_H
