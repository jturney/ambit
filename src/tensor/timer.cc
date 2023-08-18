/*
 * @BEGIN LICENSE
 *
 * ambit: C++ library for the implementation of tensor product calculations
 *        through a clean, concise user interface.
 *
 * Copyright (c) 2014-2017 Ambit developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of ambit.
 *
 * Ambit is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Ambit is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with ambit; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

//
// Created by Justin Turney on 3/30/15.
//

#include <ambit/common_types.h>
#include <ambit/settings.h>
#include <ambit/timer.h>
#include <ambit/print.h>

#include <chrono>
#include <cassert>
#include <cstring>

namespace ambit
{
namespace timer
{

using clock = std::chrono::high_resolution_clock;
using time_point = std::chrono::time_point<clock>;

namespace
{

struct TimerDetail
{
    // Description of the timing block
    string name;

    // Accumulated runtime
    clock::duration total_time;
    // Number of times the timer has been called
    size_t total_calls;

    TimerDetail *parent;
    map<string, TimerDetail> children;

    time_point start_time;

    TimerDetail()
        : name("(no name)"), total_time(0), total_calls(0), parent(nullptr)
    {
    }
};

TimerDetail *current_timer = nullptr;
TimerDetail *root = nullptr;
}

void initialize()
{
    root = new TimerDetail();
    root->name = "Total Run Time";
    root->parent = nullptr;
    root->total_calls = 1;

    current_timer = root;

    // Determine timer overhead
    for (int i = 0; i < 1000; ++i)
    {
        timer_push("Timer Overhead");
        timer_pop();
    }
}

void finalize()
{
    assert(root == current_timer);
    delete root;
    root = current_timer = nullptr;
}

namespace
{

// This is a recursive function
void print_timer_info(TimerDetail *timer)
{
    char buffer[512];
    if (timer != root)
    {
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(timer->total_time).count();
        snprintf(buffer, 512, "%lld ms : %lld calls : %lld ms per call : ",
                 time,
                 static_cast<long long>(timer->total_calls),
                 time / timer->total_calls);
        print("%s%*s%s\n", buffer,
              //              60 - ambit::current_indent() - strlen(buffer),
              60 - strlen(buffer), "", timer->name.c_str());
    }
    else
    {
        print("\nTiming information:\n\n");
    }
    if (!timer->children.empty())
    {
        indent(2);

        for (auto &child : timer->children)
        {
            print_timer_info(&child.second);
        }

        unindent(2);
    }
}
}

void report()
{
    if (settings::timers)
        print_timer_info(root);
}

void timer_push(const string &name)
{
    if (settings::timers)
    {
        assert(current_timer != nullptr);

        if (current_timer->children.count(name) == 0)
        {
            current_timer->children[name].name = name;
            current_timer->children[name].parent = current_timer;
        }

        current_timer = &current_timer->children[name];
        current_timer->start_time = clock::now();
    }
}

void timer_pop()
{
    if (settings::timers)
    {
        current_timer->total_time += clock::now() - current_timer->start_time;
        current_timer->total_calls++;

        current_timer = current_timer->parent;
    }
}
}
}
