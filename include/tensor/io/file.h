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

#if !defined(TENSOR_IO_FILE)
#define TENSOR_IO_FILE

#include <cstdint>
#include <string>
#include <vector>
#include <functional>

#if defined(DEBUG)
//#include <util/print.h>
#endif

namespace tensor { namespace io {

struct File;

enum Error {
    kIOErrorOpen = 0,
    kIOErrorReopen,
    kIOErrorClose,
    kIOErrorReclose,
    kIOErrorOStat,
    kIOErrorLSeek,
    kIOErrorRead,
    kIOErrorWrite,
    kIOErrorNoTOCEntry,
    kIOErrorTOCEntrySize,
    kIOErrorKeyLength,
    kIOErrorBlockSize,
    kIOErrorBlockStart,
    kIOErrorBlockEnd
};

// These constants are taken from PSI.
enum Constants {
    kIOMaxKeyLength = 80,
    kIOPageLength = 64*1024
};

enum OpenMode {
    kOpenModeCreateNew,
    kOpenModeOpenExisting
};

enum DeleteMode {
    kDeleteModeKeepOnClose,
    kDeleteModeDeleteOnClose
};

struct Address
{
    uint64_t page;
    uint64_t offset;
};

inline bool operator==(const Address& a1, const Address& a2)
{
    return (a1.page == a2.page &&
            a1.offset == a2.offset);
}

inline bool operator!=(const Address& a1, const Address& a2)
{
    return !(a1 == a2);
}

inline bool operator>(const Address& a1, const Address& a2)
{
    if (a1.page > a2.page)
        return true;
    else if (a1.page == a2.page && a1.offset > a2.offset)
        return true;
    return false;
}

inline bool operator<(const Address& a1, const Address& a2)
{
    if (a1.page < a2.page)
        return true;
    else if (a1.page == a2.page && a1.offset < a2.offset)
        return true;
    return false;
}

namespace util {

Address get_address(const Address& start, uint64_t shift);

/**
* @brief Given a start and end Address compute the number of bytes
* between them. Note that end denotes the beginning of the next entry
* and not the end of the current entry.
* @param start starting address
* @param end ending address
* @return the distance in bytes
*/
uint64_t get_length(const Address& start, const Address& end);

}

namespace toc {

struct Entry
{
    char key[kIOMaxKeyLength];
    Address start_address;
    Address end_address;
};

struct Manager
{
    Manager(File& owner);

    void initialize();
    void finalize();

    /// Returns the number of entries from the disk
    unsigned int size() const;

    /// Returns the number of bytes the data identified by key is.
    uint64_t size(const std::string& key);

    /// Reads the TOC entries from the file.
    void read();
    /// Writes the TOC entries to the file.
    void write() const;

    /// Print the TOC to the screen.
    void print() const;

    /**
    * @brief Does a specific key exist?
    * @param key The key to search for.
    * @return true, if found; else false.
    */
    bool exists(const std::string& key) const;
    /**
    * @brief For the given key return the entry structure.
    * @param key The key to search for
    * @return The entry struct. If not found will create new entry.
    */
    Entry& entry(const std::string& key);

    Manager(const Manager && other)
            : owner_(other.owner_), contents_(std::move(other.contents_))
    { }

private:
    uint64_t read_size() const;
    void write_size() const;

    File& owner_;
    std::vector<Entry> contents_;
};

} // namespace toc

struct File
{
    File(const std::string& full_pathname, enum OpenMode om, enum DeleteMode dm = kDeleteModeKeepOnClose);

    virtual ~File();

    /// Open a file.
    bool open(const std::string& full_pathname, enum OpenMode om);

    /// Close the file.
    void close();

    /// Access the internal handle
    int handle() const { return handle_; }

    /// Set delete mode
    void set_delete_mode(DeleteMode dm) { delete_mode_ = dm; }

    /// Access the internal TOC manager
    toc::Manager& toc() { return toc_; }
    const toc::Manager& toc() const { return toc_; }

    /** Seek to a specific Address
    */
    int seek(const Address& add) {
        return seek(add.page, add.offset);
    }

    /** Seeks to a specific page and offset in the file.
    * \param page page to go to.
    * \param offset offset in the specified page.
    */
    int seek(uint64_t page, uint64_t offset);

    /** Performs a raw read at the Address specified by add.
    * \param buffer Memory location to read into.
    * \param add Address to read from.
    * \param count Number of T's to read in.
    */
    template <typename T>
    void read(T* buffer, const Address& add, uint64_t count) {
        read_raw(buffer, add, count * sizeof(T));
    }

    /** Performs a raw write at the Address specified by add.
    * \param buffer Memory location to read into.
    * \param add Address to read from.
    * \param count Number of T's to read in.
    */
    template <typename T>
    void write(const T* buffer, const Address& add, uint64_t count) {
        write_raw(buffer, add, count * sizeof(T));
    }

    /** Performs a write of the data for an entry.
    */
    template <typename T>
    void write(const std::string& label, const T* buffer, uint64_t count) {
        // obtain or create the entry in the TOC
        toc::Entry& entry = toc_.entry(label);
        // compute location where to start the write
        Address write_start = util::get_address(entry.start_address, sizeof(toc::Entry));
        // perform a raw write at the location
        write(buffer, write_start, count);
        // update the end_address for the entry.
        entry.end_address = util::get_address(write_start, sizeof(T) * count);
    }

    /** Performs a write of the data for an entry.
    * In practice, the start_address of the entry is offset by sizeof(entry) and then the write is performed.
    * \param buffer Memory location to read into.
    * \param label Address to read from.
    */
    template <typename T>
    void write(const std::string& label, const std::vector<T>& buffer) {
        // obtain or create the entry in the TOC
        toc::Entry& entry = toc_.entry(label);
        // compute location where to start the write
        Address write_start = util::get_address(entry.start_address, sizeof(toc::Entry));
        // perform a raw write at the location
        write(buffer.data(), write_start, buffer.size());
        // update the end_address for the entry.
        entry.end_address = util::get_address(write_start, sizeof(T) * buffer.size());
    }

    /** Performs a write of the data for an entry.
    * In practice, the start_address of the entry is offset by sizeof(entry) and then the write is performed.
    * \param buffer Memory location to read into.
    * \param label Address to read from.
    * \param count Number of T's to read in.
    */
    template <typename T>
    void read(const std::string& label, T* buffer, uint64_t count) {
        // ensure the entry exists
        if (toc_.exists(label) == false)
            throw std::runtime_error("entry does not exist in the file: " + label);
        // obtain the entry (if we get here it is guaranteed to exist
        toc::Entry& entry = toc_.entry(label);
        // compute location where to start the write
        Address read_start = util::get_address(entry.start_address, sizeof(toc::Entry));
        // perform a raw write at the location
        read(buffer, read_start, count);
        // no update to the entry is performed.
        // check the read count against the end address
        Address end = util::get_address(read_start, sizeof(T) * count);
        if (end > entry.end_address)
            throw std::runtime_error("read past the end address of this entry: " + label);
    }

    /** Performs a write of the data for an entry.
    * In practice, the start_address of the entry is offset by sizeof(entry) and then the write is performed.
    * \param buffer Memory location to read into.
    * \param label Address to read into.
    */
    template <typename T>
    void read(const std::string& label, std::vector<T>& buffer) {
        // ensure the entry exists
        if (toc_.exists(label) == false)
            throw std::runtime_error("entry does not exist in the file: " + label);
        // obtain the entry (if we get here it is guaranteed to exist
        toc::Entry& entry = toc_.entry(label);
        // compute location where to start the write
        Address read_start = util::get_address(entry.start_address, sizeof(toc::Entry));
        // perform a raw write at the location
        read(buffer.data(), read_start, buffer.size());
        // no update to the entry is performed.
        // check the read count against the end address
        Address end = util::get_address(read_start, sizeof(T) * buffer.size());
        if (end > entry.end_address)
            throw std::runtime_error("read past the end address of this entry: " + label);
    }

    /** Performs a write of the data for an entry.
    * In practice, the start_address of the entry is offset by sizeof(entry) and then the write is performed.
    * \param buffer Memory location to read into.
    * \param entry Address to read from.
    * \param count Number of T's to read in.
    */
    template <typename T>
    void write(toc::Entry& entry, const T* buffer, uint64_t count) {
        // compute location where to start the write
        Address write_start = util::get_address(entry.start_address, sizeof(toc::Entry));
        // perform a raw write at the location
        write(buffer, write_start, count);
        // update the end_address for the entry.
        entry.end_address = util::get_address(write_start, sizeof(T) * count);
    }

    /** Performs a write of the data for an entry.
    * In practice, the start_address of the entry is offset by sizeof(entry) and then the write is performed.
    * \param buffer Memory location to read into.
    * \param entry Address to read from.
    * \param count Number of T's to read in.
    */
    template <typename T>
    void read(const toc::Entry& entry, T* buffer, uint64_t count) {
        // compute location where to start the write
        Address read_start = util::get_address(entry.start_address, sizeof(toc::Entry));
        // perform a raw write at the location
        read(buffer, read_start, count);
        // no update to the entry is performed.
    }

    /** Performs a write of the data for an entry.
    * In practice, the start_address of the entry is offset by sizeof(entry) and then the write is performed.
    * \param buffer Memory location to read into.
    * \param entry TOC entry read from.
    */
    template <typename T>
    void read(const toc::Entry& entry, std::vector<T>& buffer) {
        // compute location where to start the write
        Address read_start = util::get_address(entry.start_address, sizeof(toc::Entry));
        // perform a raw write at the location
        read(buffer.data(), read_start, buffer.size());
        // no update to the entry is performed.
    }

    /** Performs a streaming write of the data for an entry.
    * In practice, the end_address is the start of the write.
    * \param buffer Memory location to read into.
    * \param entry Address to read from.
    * \param count Number of T's to read in.
    */
    template <typename T>
    void write_stream(toc::Entry& entry, const T* buffer, uint64_t count) {
        // compute location where to start the write
        Address write_start = util::get_address(entry.end_address, sizeof(toc::Entry));
        // perform a raw write at the location
        write(buffer, write_start, count);
        // update the end_address for the entry.
        entry.end_address = util::get_address(write_start, sizeof(T) * count);
    }

    /** Performs a streaming read of the data for an entry.
    * \param buffer Memory location to read into.
    * \param entry Address to read from.
    * \param count Number of T's to read in.
    */
    template <typename T>
    void read_stream(const toc::Entry& entry, Address & next, T* buffer, uint64_t count) {
        if (next.page == 0 && next.offset == 0)
            next = util::get_address(entry.start_address, sizeof(toc::Entry));

        // compute location where to start the write
        Address read_start = next;
        // perform a raw write at the location
        read(buffer, read_start, count);
        // update the end_address for the entry.
        next = util::get_address(read_start, sizeof(T) * count);

        if (next > entry.end_address)
            throw std::runtime_error("read_stream: read beyond the extend of the entry.");
    }

    /** Performs a streaming read of the data for the entry.
    * \param label Entry to read.
    * \param next Address to read from.
    * \param buffer Where to place the data.
    * \param count How many units of the buffer to read.
    */
    template <typename T>
    void read_entry_stream(const std::string& label, Address & next, T* buffer, uint64_t count) {
        // ensure the label exists
        if (toc_.exists(label) == false)
            throw std::runtime_error("entry does not exist in the file: " + label);
        // obtain the entry (if we get here it will exist
        toc::Entry& entry = toc_.entry(label);
        // our first time in the call, initialize next
        if (next.page == 0 && next.offset == 0) {

            // compute location where to start the write
            next = util::get_address(entry.start_address, sizeof(toc::Entry));
        }

        // perform read
        read(buffer, next, count);
        // shift next over
        next = util::get_address(next, sizeof(T) * count);
        if (next > entry.end_address)
            throw std::runtime_error("read past the end address of this entry: " + label);
    }

//    file& operator=(file&& other)
//    {
//        handle_ = std::move(other.handle_);
//        name_   = std::move(other.name_);
//        read_stat_ = std::move(other.read_stat_);
//        write_stat_ = std::move(other.write_stat_);
//        toc_ = std::move(other.toc_);
//        return *this;
//    }

    File(File&& other)
            : handle_(other.handle_), name_(other.name_), read_stat_(other.read_stat_), write_stat_(other.write_stat_), toc_(std::move(other.toc_)), open_mode_(std::move(other.open_mode_)), delete_mode_(std::move(other.delete_mode_))
    {
        other.handle_ = -1;
    }

protected:

    /** Performs the ultimate reading from the file. Will seek to add and read size number of bytes into buffer.
    */
    void read_raw(void *buffer, const Address & add, uint64_t size);

    /** Performs the ultimate writing to the file. Will seek to add and write size number of bytes from buffer.
    */
    void write_raw(const void *buffer, const Address & add, uint64_t size);

    /** Used internally to report an error to the user.
    */
    void error(Error code);

    /// low-level file handle
    int handle_;

    /// the name of the file
    std::string name_;

    uint64_t read_stat_;
    uint64_t write_stat_;

    toc::Manager toc_;

    const OpenMode open_mode_;
    DeleteMode delete_mode_;

    friend struct toc::Manager;
};

}}

#endif
