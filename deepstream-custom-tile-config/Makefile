
APP:= deepstream-custom-tile-config

SRCS:= $(wildcard *.c)

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0 gstreamer-video-1.0 glib-2.0

OBJS:= $(SRCS:.c=.o)

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -lm -L/usr/local/cuda/lib64/ -lcudart
CFLAGS+= -I/usr/local/cuda/include

all: $(APP)

.o: .c $(INCS) Makefile
	$(CC) -c $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CC) -o $(APP) $(OBJS) $(LIBS)

clean:
	rm -rf $(OBJS) $(APP)

