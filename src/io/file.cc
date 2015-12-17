/*
 *  Copyright (C) 2013  Justin Turney
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.

 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.

 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-151 USA.
 */

#include <ambit/io/file.h>

#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cerrno>
#include <cstring>

//#include <util/print.h>

namespace ambit
{
namespace io
{

namespace util
{

Address get_address(const Address &start, uint64_t shift)
{
    Address new_address;
    uint64_t bytes_left;

    bytes_left = kIOPageLength - start.offset;

    if (shift >= bytes_left)
    { // shift to later page
        new_address.page =
            start.page + (shift - bytes_left) / kIOPageLength + 1;
        new_address.offset =
            shift - bytes_left -
            (new_address.page - start.page - 1) * kIOPageLength;
    }
    else
    { // block starts on current page
        new_address.page = start.page;
        new_address.offset = start.offset + shift;
    }

    return new_address;
}

uint64_t get_length(const Address &start, const Address &end)
{
    uint64_t full_page_bytes;

    full_page_bytes = (end.page - start.page - 1) * kIOPageLength;

    if (start.page == end.page)
        return end.offset - start.offset;
    else
        return ((kIOPageLength - start.offset) + full_page_bytes + end.offset);
}
}

namespace toc
{

Manager::Manager(File &owner) : owner_(owner) {}

void Manager::initialize() {}

void Manager::finalize() { write(); }

unsigned int Manager::size() const { return contents_.size(); }

bool Manager::exists(const std::string &key) const
{
    for (const Entry &e : contents_)
    {
        if (key == e.key)
            return true;
    }
    return false;
}

Entry &Manager::entry(const std::string &key)
{
    for (Entry &e : contents_)
    {
        if (key == e.key)
            return e;
    }

    // if we get here then we didn't find it.
    // take the last entry and use its end address
    // as our start address.

    // handle special case of no entries.
    Entry e;
    ::strcpy(e.key, key.c_str());
    if (contents_.size() == 0)
        e.start_address = {0, sizeof(uint64_t)};
    else
        e.start_address = contents_.end()->end_address;
    e.end_address = util::get_address(e.start_address, sizeof(Entry));

    contents_.push_back(e);
    return contents_.back();
}

uint64_t Manager::size(const std::string &key)
{
    if (exists(key))
    {
        const Entry &found = entry(key);
        return util::get_length(found.start_address, found.end_address) -
               sizeof(Entry);
    }

    return 0;
}

uint64_t Manager::read_size() const
{
    const int handle = owner_.handle();
    int error_code;
    uint64_t len;

    error_code = ::lseek(handle, 0L, SEEK_SET);
    if (error_code == -1)
        owner_.error(kIOErrorLSeek);

    error_code = ::read(handle, &len, sizeof(uint64_t));

    if (error_code != sizeof(uint64_t))
        len = 0;

    //    ambit::util::print0("read_length: length %lu\n", len);

    return len;
}

void Manager::write_size() const
{
    const int handle = owner_.handle();
    int error_code;

    error_code = ::lseek(handle, 0L, SEEK_SET);
    if (error_code == -1)
        owner_.error(kIOErrorLSeek);

    uint64_t len = contents_.size();
    error_code = ::write(handle, &len, sizeof(uint64_t));
    if (error_code != sizeof(uint64_t))
        owner_.error(kIOErrorWrite);
}

void Manager::read()
{
    uint64_t len = read_size();

    // clear out existing vector
    contents_.clear();
    if (len)
    {
        Address zero = {0, 0};
        Address add;

        // start one uint64_t from the start of the file
        add = util::get_address(zero, sizeof(uint64_t));
        for (uint64_t i = 0; i < len; ++i)
        {
            Entry new_entry;
            owner_.read(&new_entry, add, 1);

            contents_.push_back(new_entry);
            add = new_entry.end_address;
        }
    }
}

void Manager::write() const
{
    write_size();

    for (const Entry &e : contents_)
    {
        owner_.write(&e, e.start_address, 1);
    }
}

void Manager::print() const
{
    printf("-------------------------------------------------------------------"
           "---------\n");
    printf("Key                                   Spage    Soffset      Epage  "
           "  Eoffset\n");
    printf("-------------------------------------------------------------------"
           "---------\n");

    for (const Entry &e : contents_)
    {
        printf("%-32s %10zu %10zu %10zu %10zu\n", e.key, e.start_address.page,
               e.start_address.offset, e.end_address.page,
               e.end_address.offset);
    }
}

} // namespace toc

File::File(const std::string &full_pathname, enum OpenMode om,
           enum DeleteMode dm)
    : handle_(-1), name_(full_pathname), read_stat_(0), write_stat_(0),
      toc_(*this), open_mode_(om), delete_mode_(dm)
{
    if (open(full_pathname, om) == false)
        throw std::runtime_error("file: Unable to open file " + name_);
}

File::~File() { close(); }

bool File::open(const std::string &full_pathname, enum OpenMode om)
{
    if (handle_ != -1)
        return false;

    name_ = full_pathname;
    if (om == kOpenModeOpenExisting)
    {
        handle_ = ::open(full_pathname.c_str(), O_CREAT | O_RDWR, 0644);
        if (handle_ == -1)
            throw std::runtime_error("unable to open file: " + full_pathname);
        toc_.read();
    }
    else
    {
        handle_ =
            ::open(full_pathname.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
        if (handle_ == -1)
            throw std::runtime_error("unable to open file: " + full_pathname);
        toc_.initialize();
    }

    if (handle_ == -1) // error occurred
        return false;
    return true;
}

void File::close()
{
    if (handle_ != -1)
    {
        toc_.finalize();
        ::close(handle_);

        if (delete_mode_ == kDeleteModeDeleteOnClose)
            ::unlink(name_.c_str());
    }
    handle_ = -1;
}

void File::error(Error code)
{
    static const char *error_message[] = {
        "file not open or open call failed",
        "file is already open",
        "file close failed",
        "file is already closed",
        "invalid status flag for file open",
        "lseek failed",
        "error reading from file",
        "error writing to file",
        "no such TOC entry",
        "TOC entry size mismatch",
        "TOC key too long",
        "requested block size is invalid",
        "incorrect block start address",
        "incorrect block end address",
    };

    printf("io error: %d, %s; errno %d\n", code, error_message[code], errno);

    ::exit(EXIT_FAILURE);
}

int File::seek(uint64_t page, uint64_t offset)
{
    // this is strictly to avoid overflow errors on lseek calls
    const uint64_t bignum = 10000;
    int error_code;
    uint64_t total_offset;

    // move to the beginning
    error_code = ::lseek(handle_, 0, SEEK_SET);

    if (error_code == -1)
        return error_code;

    // lseek through large chunks of the file to avoid offset overflows
    for (; page > bignum; page -= bignum)
    {
        total_offset = bignum * kIOPageLength;
        error_code = ::lseek(handle_, total_offset, SEEK_CUR);
        if (error_code == -1)
            return error_code;
    }

    // now compute the file offset including the page-relative term
    total_offset = page;
    total_offset *= kIOPageLength;
    total_offset += offset; // add the page-relative term

    //    ambit::util::print0("seeking to %lu (page %lu, offset %lu, page length
    //    %lu)\n",
    //                        total_offset, page, offset, kIOPageLength);

    error_code = ::lseek(handle_, total_offset, SEEK_CUR);
    if (error_code == -1)
        return error_code;

    return 0;
}

void File::read_raw(void *buffer, const Address &add, uint64_t size)
{
    uint64_t error_code;

    // seek to the needed address
    seek(add);

    error_code = ::read(handle_, buffer, size);
    if (error_code != size)
    {
        printf("size = %zu error_code %zu\n", size, error_code);
        error(kIOErrorRead);
    }
    read_stat_ += size;
}

void File::write_raw(const void *buffer, const Address &add, uint64_t size)
{
    uint64_t error_code;

    // seek to the needed address
    seek(add);

    error_code = ::write(handle_, buffer, size);
    if (error_code != size)
        error(kIOErrorRead);

    write_stat_ += size;
}
}
}
