#include <ambit/print.h>
#include <ambit/settings.h>
#include <ambit/tensor.h>

#include <cstdarg>

namespace ambit
{

namespace
{

int indent_size = 0;

void print_indentation() { printf("%*s", indent_size, ""); }
}

int current_indent() { return indent_size; }

void indent(int increment) { indent_size += increment; }

void unindent(int decrement)
{
    indent_size -= decrement;
    if (indent_size < 0)
        indent_size = 0;
}

void print(const string &format, ...)
{
    if (ambit::settings::rank == 0)
    {
        va_list args;
        va_start(args, format);
        print_indentation();
        vprintf(format.c_str(), args);
        va_end(args);
    }
}

void printn(const string &format, ...)
{
    for (int proc = 0; proc < settings::nprocess; proc++)
    {
        if (proc == settings::rank)
        {
            printf("%d: ", settings::rank);
            va_list args;
            va_start(args, format);
            print_indentation();
            vprintf(format.c_str(), args);
            va_end(args);
        }

        barrier();
    }
}
}
