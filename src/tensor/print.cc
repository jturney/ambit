#include <ambit/print.h>

#include <cstdio>
#include <cstdarg>
#include <ambit/tensor.h>

namespace ambit {

namespace {
int indent_size = 0;


void print_indentation(FILE *out)
{
    fprintf(out, "%*s", indent_size, "");
}

}

void indent(int increment)
{
    indent_size += increment;
}

void unindent(int decrement)
{
    indent_size -= decrement;
    if (indent_size < 0)
        indent_size = 0;
}

void print(const std::string& format, ...)
{
    if (ambit::settings::rank == 0) {
        va_list args;
        va_start(args, format);
        print_indentation(stdout);
        vprintf(format.c_str(), args);
        va_end(args);
    }
}

//void printn(const std::string& format, ...)
//{
//    throw std::runtime_error("Not implemented");
//}

}
