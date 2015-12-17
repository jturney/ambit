//
// Created by Justin Turney on 3/30/15.
//

#ifndef AMBIT_TIMER_H
#define AMBIT_TIMER_H

#include "common_types.h"

namespace ambit
{
namespace timer
{

// TODO: Should be hidden from public interface
void initialize();

// TODO: Should be hidden from public interface
void finalize();

void report();

void timer_push(const string &name);
void timer_pop();
}
}

#endif // AMBIT_TIMER_H
