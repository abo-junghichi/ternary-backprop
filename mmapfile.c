#ifndef MMAPFILE_C
#define MMAPFILE_C
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
typedef struct {
    void *addr;
    size_t length;
} mmapfile;
static mmapfile mmapfile_open(const char *pathname, int for_read,
			      int for_write, size_t length)
{
    mmapfile rtn;
    int open_flag = for_write ? O_RDWR : O_RDONLY;
    int mmap_flags =
	(for_read ? PROT_READ : 0) | (for_write ? PROT_WRITE : 0);
    int fd;
    if (0 == mmap_flags) {
	rtn.addr = NULL;
	rtn.length = 0;
	return rtn;
    }
    fd = open(pathname, open_flag);
    rtn.addr = mmap(NULL, length, mmap_flags, MAP_SHARED, fd, 0);
    rtn.length = length;
    close(fd);
    return rtn;
}
static int mmapfile_close(mmapfile mf)
{
    return munmap(mf.addr, mf.length);
}
#endif				/* MMAPFILE_C */
