#if !defined(TENSOR_INCLUDE_PRINT_H)
#define TENSOR_INCLUDE_PRINT_H

#include <string>

namespace ambit {

/// Print function. Only the master node is allowed to print to the screen.
void print(const std::string& format, ...);

/** Each proces will print to their respective output file.
 *  The master process will print to both the screen and its output file.
 */
//void printn(const std::string& format, ...);

/** Increases printing column offset by increment.
 * @param increment the amount to increase indentation.
 */
void indent(int increment = 4);

/** Decreases printing column offset by increment.
 * @param decrement the amount the decrease indentation
 */
void unindent(int decrement = 4);

struct indenter
{
    indenter(int increment = 4) : size(increment) { indent(size); }
    ~indenter() { unindent(size); }

private:
    int size;
};

}

#endif // TENSOR_INCLUDE_PRINT_H
