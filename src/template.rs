pub mod solution {
    use crate::algo_lib::collections::btree_ext::BTreeExt;
    use crate::algo_lib::collections::default_map::default_hash_map::DefaultHashMap;
    use crate::algo_lib::collections::slice_ext::indices::Indices;
    use crate::algo_lib::io::input::Input;
    use crate::algo_lib::io::output::Output;
    use std::collections::BTreeSet;

    type PreCalc = ();

    fn solve(input: &mut Input, out: &mut Output, _test_case: usize, _data: &PreCalc) {
        let n = input.read_size();
        let q = input.read_size();
        let a = input.read_unsigned_vec(n);

        let mut xor = 0;
        let mut x = Vec::with_capacity(n + 1);
        let mut pos = DefaultHashMap::<_, BTreeSet<_>>::new();
        for i in a.indices() {
            x.push(xor);
            pos[xor].insert(i);
            xor ^= a[i];
        }
        x.push(xor);
        pos[xor].insert(n);

        for _ in 0..q {
            let l = input.read_size() - 1;
            let r = input.read_size();

            if x[l] == x[r] {
                out.print_line(true);
                continue;
            }
            let last_l = pos[x[l]].floor(&r).unwrap();
            let first_r = pos[x[r]].ceil(&l).unwrap();
            out.print_line(last_l > first_r);
        }
    }

    pub(crate) fn run(mut input: Input, mut output: Output) -> bool {
        let pre_calc = ();

        #[allow(dead_code)]
        enum TestType {
            Single,
            MultiNumber,
            MultiEof,
        }
        let test_type = TestType::MultiNumber;
        match test_type {
            TestType::Single => solve(&mut input, &mut output, 1, &pre_calc),
            TestType::MultiNumber => {
                let t = input.read();
                for i in 1..=t {
                    solve(&mut input, &mut output, i + 1, &pre_calc);
                }
            }
            TestType::MultiEof => {
                let mut i = 1;
                while input.peek().is_some() {
                    solve(&mut input, &mut output, i, &pre_calc);
                    i += 1;
                }
            }
        }
        output.flush();
        input.skip_whitespace();
        input.peek().is_none()
    }

}
pub mod algo_lib {
    pub mod collections {
        pub mod btree_ext {
            use std::collections::BTreeMap;
            use std::collections::BTreeSet;
            use std::ops::Bound;

            pub trait BTreeExt<'a, T> {
                type Output;
                fn next(&'a self, x: &'a T) -> Option<Self::Output>;
                fn prev(&'a self, x: &'a T) -> Option<Self::Output>;
                fn ceil(&'a self, x: &'a T) -> Option<Self::Output>;
                fn floor(&'a self, x: &'a T) -> Option<Self::Output>;
            }

            impl<'a, T: Ord + 'a> BTreeExt<'a, T> for BTreeSet<T> {
                type Output = &'a T;
                fn next(&'a self, x: &'a T) -> Option<Self::Output> {
                    self.range((Bound::Excluded(x), Bound::Unbounded)).next()
                }

                fn ceil(&'a self, x: &'a T) -> Option<Self::Output> {
                    self.range(x..).next()
                }

                fn prev(&'a self, x: &'a T) -> Option<Self::Output> {
                    self.range(..x).next_back()
                }

                fn floor(&'a self, x: &'a T) -> Option<Self::Output> {
                    self.range(..=x).next_back()
                }
            }

            impl<'a, K: Ord + 'a, V: 'a> BTreeExt<'a, K> for BTreeMap<K, V> {
                type Output = (&'a K, &'a V);

                fn next(&'a self, x: &'a K) -> Option<Self::Output> {
                    self.range((Bound::Excluded(x), Bound::Unbounded)).next()
                }

                fn ceil(&'a self, x: &'a K) -> Option<Self::Output> {
                    self.range(x..).next()
                }

                fn prev(&'a self, x: &'a K) -> Option<Self::Output> {
                    self.range(..x).next_back()
                }

                fn floor(&'a self, x: &'a K) -> Option<Self::Output> {
                    self.range(..=x).next_back()
                }
            }
        }
        pub mod default_map {
            pub mod default_hash_map {
                use std::collections::HashMap;
                use std::hash::Hash;
                use std::iter::FromIterator;
                use std::ops::Deref;
                use std::ops::DerefMut;
                use std::ops::Index;
                use std::ops::IndexMut;

                #[derive(Default, Clone, Eq, PartialEq)]
                pub struct DefaultHashMap<K: Hash + Eq, V>(HashMap<K, V>, V);

                impl<K: Hash + Eq, V> Deref for DefaultHashMap<K, V> {
                    type Target = HashMap<K, V>;

                    fn deref(&self) -> &Self::Target {
                        &self.0
                    }
                }

                impl<K: Hash + Eq, V> DerefMut for DefaultHashMap<K, V> {
                    fn deref_mut(&mut self) -> &mut Self::Target {
                        &mut self.0
                    }
                }

                impl<K: Hash + Eq, V: Default> DefaultHashMap<K, V> {
                    pub fn new() -> Self {
                        Self(HashMap::new(), V::default())
                    }

                    pub fn with_capacity(cap: usize) -> Self {
                        Self(HashMap::with_capacity(cap), V::default())
                    }

                    pub fn get(&self, key: &K) -> &V {
                        self.0.get(key).unwrap_or(&self.1)
                    }

                    pub fn get_mut(&mut self, key: K) -> &mut V {
                        self.0.entry(key).or_insert_with(|| V::default())
                    }

                    pub fn into_values(self) -> std::collections::hash_map::IntoValues<K, V> {
                        self.0.into_values()
                    }
                }

                impl<K: Hash + Eq, V: Default> Index<K> for DefaultHashMap<K, V> {
                    type Output = V;

                    fn index(&self, index: K) -> &Self::Output {
                        self.get(&index)
                    }
                }

                impl<K: Hash + Eq, V: Default> IndexMut<K> for DefaultHashMap<K, V> {
                    fn index_mut(&mut self, index: K) -> &mut Self::Output {
                        self.get_mut(index)
                    }
                }

                impl<K: Hash + Eq, V> IntoIterator for DefaultHashMap<K, V> {
                    type Item = (K, V);
                    type IntoIter = std::collections::hash_map::IntoIter<K, V>;

                    fn into_iter(self) -> Self::IntoIter {
                        self.0.into_iter()
                    }
                }

                impl<K: Hash + Eq, V: Default> FromIterator<(K, V)> for DefaultHashMap<K, V> {
                    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
                        Self(iter.into_iter().collect(), V::default())
                    }
                }
            }
        }
        pub mod slice_ext {
            pub mod indices {
                use std::ops::Range;

                pub trait Indices {
                    fn indices(&self) -> Range<usize>;
                }

                impl<T> Indices for [T] {
                    fn indices(&self) -> Range<usize> {
                        0..self.len()
                    }
                }
            }
        }
        pub mod vec_ext {
            pub mod default {
                pub fn default_vec<T: Default>(len: usize) -> Vec<T> {
                    let mut v = Vec::with_capacity(len);
                    for _ in 0..len {
                        v.push(T::default());
                    }
                    v
                }
            }
        }
    }
    pub mod io {
        pub mod input {
            use crate::algo_lib::collections::vec_ext::default::default_vec;
            use std::io::Read;

            pub struct Input<'s> {
                input: &'s mut dyn Read,
                buf: Vec<u8>,
                at: usize,
                buf_read: usize,
            }

            macro_rules! read_impl {
($t: ty, $read_name: ident, $read_vec_name: ident) => {
pub fn $read_name(&mut self) -> $t {
self.read()
}

pub fn $read_vec_name(&mut self, len: usize) -> Vec<$t> {
self.read_vec(len)
}
};

($t: ty, $read_name: ident, $read_vec_name: ident, $read_pair_vec_name: ident) => {
read_impl!($t, $read_name, $read_vec_name);

pub fn $read_pair_vec_name(&mut self, len: usize) -> Vec<($t, $t)> {
self.read_vec(len)
}
};
}

            impl<'s> Input<'s> {
                const DEFAULT_BUF_SIZE: usize = 4096;

                pub fn new(input: &'s mut dyn Read) -> Self {
                    Self {
                        input,
                        buf: default_vec(Self::DEFAULT_BUF_SIZE),
                        at: 0,
                        buf_read: 0,
                    }
                }

                pub fn new_with_size(input: &'s mut dyn Read, buf_size: usize) -> Self {
                    Self {
                        input,
                        buf: default_vec(buf_size),
                        at: 0,
                        buf_read: 0,
                    }
                }

                pub fn get(&mut self) -> Option<u8> {
                    if self.refill_buffer() {
                        let res = self.buf[self.at];
                        self.at += 1;
                        if res == b'\r' {
                            if self.refill_buffer() && self.buf[self.at] == b'\n' {
                                self.at += 1;
                            }
                            return Some(b'\n');
                        }
                        Some(res)
                    } else {
                        None
                    }
                }

                pub fn peek(&mut self) -> Option<u8> {
                    if self.refill_buffer() {
                        let res = self.buf[self.at];
                        Some(if res == b'\r' { b'\n' } else { res })
                    } else {
                        None
                    }
                }

                pub fn skip_whitespace(&mut self) {
                    while let Some(b) = self.peek() {
                        if !char::from(b).is_whitespace() {
                            return;
                        }
                        self.get();
                    }
                }

                pub fn next_token(&mut self) -> Option<Vec<u8>> {
                    self.skip_whitespace();
                    let mut res = Vec::new();
                    while let Some(c) = self.get() {
                        if char::from(c).is_whitespace() {
                            break;
                        }
                        res.push(c);
                    }
                    if res.is_empty() {
                        None
                    } else {
                        Some(res)
                    }
                }

                //noinspection RsSelfConvention
                pub fn is_exhausted(&mut self) -> bool {
                    self.peek().is_none()
                }

                //noinspection RsSelfConvention
                pub fn is_empty(&mut self) -> bool {
                    self.skip_whitespace();
                    self.is_exhausted()
                }

                pub fn read<T: Readable>(&mut self) -> T {
                    T::read(self)
                }

                pub fn read_vec<T: Readable>(&mut self, size: usize) -> Vec<T> {
                    let mut res = Vec::with_capacity(size);
                    for _ in 0..size {
                        res.push(self.read());
                    }
                    res
                }

                pub fn read_char(&mut self) -> char {
                    self.skip_whitespace();
                    self.get().unwrap().into()
                }

                read_impl!(u32, read_unsigned, read_unsigned_vec);
                read_impl!(u64, read_u64, read_u64_vec);
                read_impl!(usize, read_size, read_size_vec, read_size_pair_vec);
                read_impl!(i32, read_int, read_int_vec, read_int_pair_vec);
                read_impl!(i64, read_long, read_long_vec, read_long_pair_vec);
                read_impl!(i128, read_i128, read_i128_vec);

                fn refill_buffer(&mut self) -> bool {
                    if self.at == self.buf_read {
                        self.at = 0;
                        self.buf_read = self.input.read(&mut self.buf).unwrap();
                        self.buf_read != 0
                    } else {
                        true
                    }
                }
            }

            pub trait Readable {
                fn read(input: &mut Input) -> Self;
            }

            impl Readable for char {
                fn read(input: &mut Input) -> Self {
                    input.read_char()
                }
            }

            impl<T: Readable> Readable for Vec<T> {
                fn read(input: &mut Input) -> Self {
                    let size = input.read();
                    input.read_vec(size)
                }
            }

            macro_rules! read_integer {
($($t:ident)+) => {$(
impl Readable for $t {
fn read(input: &mut Input) -> Self {
input.skip_whitespace();
let mut c = input.get().unwrap();
let sgn = match c {
b'-' => {
c = input.get().unwrap();
true
}
b'+' => {
c = input.get().unwrap();
false
}
_ => false,
};
let mut res = 0;
loop {
assert!(c.is_ascii_digit());
res *= 10;
let d = (c - b'0') as $t;
if sgn {
res -= d;
} else {
res += d;
}
match input.get() {
None => break,
Some(ch) => {
if ch.is_ascii_whitespace() {
break;
} else {
c = ch;
}
}
}
}
res
}
}
)+};
}

            read_integer!(i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize);

            macro_rules! tuple_readable {
($($name:ident)+) => {
impl<$($name: Readable), +> Readable for ($($name,)+) {
fn read(input: &mut Input) -> Self {
($($name::read(input),)+)
}
}
}
}

            tuple_readable! {T}
            tuple_readable! {T U}
            tuple_readable! {T U V}
            tuple_readable! {T U V X}
            tuple_readable! {T U V X Y}
            tuple_readable! {T U V X Y Z}
            tuple_readable! {T U V X Y Z A}
            tuple_readable! {T U V X Y Z A B}
            tuple_readable! {T U V X Y Z A B C}
            tuple_readable! {T U V X Y Z A B C D}
            tuple_readable! {T U V X Y Z A B C D E}
            tuple_readable! {T U V X Y Z A B C D E F}

            impl Read for Input<'_> {
                fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                    if self.at == self.buf_read {
                        self.input.read(buf)
                    } else {
                        let mut i = 0;
                        while i < buf.len() && self.at < self.buf_read {
                            buf[i] = self.buf[self.at];
                            i += 1;
                            self.at += 1;
                        }
                        Ok(i)
                    }
                }
            }
        }
        pub mod output {
            use crate::algo_lib::collections::vec_ext::default::default_vec;
            use std::cmp::Reverse;
            use std::io::stderr;
            use std::io::Stderr;
            use std::io::Write;

            #[derive(Copy, Clone)]
            pub enum BoolOutput {
                YesNo,
                YesNoCaps,
                PossibleImpossible,
                Custom(&'static str, &'static str),
            }

            impl BoolOutput {
                pub fn output(&self, output: &mut Output, val: bool) {
                    (if val { self.yes() } else { self.no() }).write(output);
                }

                fn yes(&self) -> &str {
                    match self {
                        BoolOutput::YesNo => "Yes",
                        BoolOutput::YesNoCaps => "YES",
                        BoolOutput::PossibleImpossible => "Possible",
                        BoolOutput::Custom(yes, _) => yes,
                    }
                }

                fn no(&self) -> &str {
                    match self {
                        BoolOutput::YesNo => "No",
                        BoolOutput::YesNoCaps => "NO",
                        BoolOutput::PossibleImpossible => "Impossible",
                        BoolOutput::Custom(_, no) => no,
                    }
                }
            }

            pub struct Output<'s> {
                output: &'s mut dyn Write,
                buf: Vec<u8>,
                at: usize,
                auto_flush: bool,
                bool_output: BoolOutput,
            }

            impl<'s> Output<'s> {
                const DEFAULT_BUF_SIZE: usize = 4096;

                pub fn new(output: &'s mut dyn Write) -> Self {
                    Self {
                        output,
                        buf: default_vec(Self::DEFAULT_BUF_SIZE),
                        at: 0,
                        auto_flush: false,
                        bool_output: BoolOutput::YesNoCaps,
                    }
                }

                pub fn new_with_auto_flush(output: &'s mut dyn Write) -> Self {
                    Self {
                        output,
                        buf: default_vec(Self::DEFAULT_BUF_SIZE),
                        at: 0,
                        auto_flush: true,
                        bool_output: BoolOutput::YesNoCaps,
                    }
                }

                pub fn flush(&mut self) {
                    if self.at != 0 {
                        self.output.write_all(&self.buf[..self.at]).unwrap();
                        self.output.flush().unwrap();
                        self.at = 0;
                    }
                }

                pub fn print<T: Writable>(&mut self, s: T) {
                    s.write(self);
                    self.maybe_flush();
                }

                pub fn print_line<T: Writable>(&mut self, s: T) {
                    self.print(s);
                    self.put(b'\n');
                    self.maybe_flush();
                }

                pub fn put(&mut self, b: u8) {
                    self.buf[self.at] = b;
                    self.at += 1;
                    if self.at == self.buf.len() {
                        self.flush();
                    }
                }

                pub fn maybe_flush(&mut self) {
                    if self.auto_flush {
                        self.flush();
                    }
                }

                pub fn print_per_line<T: Writable>(&mut self, arg: &[T]) {
                    for i in arg {
                        i.write(self);
                        self.put(b'\n');
                    }
                }

                pub fn print_iter<T: Writable, I: Iterator<Item = T>>(&mut self, iter: I) {
                    let mut first = true;
                    for e in iter {
                        if first {
                            first = false;
                        } else {
                            self.put(b' ');
                        }
                        e.write(self);
                    }
                }

                pub fn set_bool_output(&mut self, bool_output: BoolOutput) {
                    self.bool_output = bool_output;
                }
            }

            impl Write for Output<'_> {
                fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                    let mut start = 0usize;
                    let mut rem = buf.len();
                    while rem > 0 {
                        let len = (self.buf.len() - self.at).min(rem);
                        self.buf[self.at..self.at + len].copy_from_slice(&buf[start..start + len]);
                        self.at += len;
                        if self.at == self.buf.len() {
                            self.flush();
                        }
                        start += len;
                        rem -= len;
                    }
                    self.maybe_flush();
                    Ok(buf.len())
                }

                fn flush(&mut self) -> std::io::Result<()> {
                    self.flush();
                    Ok(())
                }
            }

            pub trait Writable {
                fn write(&self, output: &mut Output);
            }

            impl Writable for &str {
                fn write(&self, output: &mut Output) {
                    output.write_all(self.as_bytes()).unwrap();
                }
            }

            impl Writable for String {
                fn write(&self, output: &mut Output) {
                    output.write_all(self.as_bytes()).unwrap();
                }
            }

            impl Writable for char {
                fn write(&self, output: &mut Output) {
                    output.put(*self as u8);
                }
            }

            impl<T: Writable> Writable for [T] {
                fn write(&self, output: &mut Output) {
                    output.print_iter(self.iter());
                }
            }

            impl<T: Writable, const N: usize> Writable for [T; N] {
                fn write(&self, output: &mut Output) {
                    output.print_iter(self.iter());
                }
            }

            impl<T: Writable> Writable for &T {
                fn write(&self, output: &mut Output) {
                    T::write(self, output)
                }
            }

            impl<T: Writable> Writable for Vec<T> {
                fn write(&self, output: &mut Output) {
                    self.as_slice().write(output);
                }
            }

            impl Writable for () {
                fn write(&self, _output: &mut Output) {}
            }

            macro_rules! write_to_string {
($($t:ident)+) => {$(
impl Writable for $t {
fn write(&self, output: &mut Output) {
self.to_string().write(output);
}
}
)+};
}

            write_to_string!(u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize);

            macro_rules! tuple_writable {
($name0:ident $($name:ident: $id:tt )*) => {
impl<$name0: Writable, $($name: Writable,)*> Writable for ($name0, $($name,)*) {
fn write(&self, out: &mut Output) {
self.0.write(out);
$(
out.put(b' ');
self.$id.write(out);
)*
}
}
}
}

            tuple_writable! {T}
            tuple_writable! {T U:1}
            tuple_writable! {T U:1 V:2}
            tuple_writable! {T U:1 V:2 X:3}
            tuple_writable! {T U:1 V:2 X:3 Y:4}
            tuple_writable! {T U:1 V:2 X:3 Y:4 Z:5}
            tuple_writable! {T U:1 V:2 X:3 Y:4 Z:5 A:6}
            tuple_writable! {T U:1 V:2 X:3 Y:4 Z:5 A:6 B:7}
            tuple_writable! {T U:1 V:2 X:3 Y:4 Z:5 A:6 B:7 C:8}

            impl<T: Writable> Writable for Option<T> {
                fn write(&self, output: &mut Output) {
                    match self {
                        None => (-1).write(output),
                        Some(t) => t.write(output),
                    }
                }
            }

            impl Writable for bool {
                fn write(&self, output: &mut Output) {
                    let bool_output = output.bool_output;
                    bool_output.output(output, *self)
                }
            }

            impl<T: Writable> Writable for Reverse<T> {
                fn write(&self, output: &mut Output) {
                    self.0.write(output);
                }
            }

            static mut ERR: Option<Stderr> = None;

            pub fn err() -> Output<'static> {
                unsafe {
                    if ERR.is_none() {
                        ERR = Some(stderr());
                    }
                    Output::new_with_auto_flush(ERR.as_mut().unwrap())
                }
            }
        }
    }
}
fn main() {
    let mut sin = std::io::stdin();
    let input = algo_lib::io::input::Input::new(&mut sin);
    let mut stdout = std::io::stdout();
    let output = algo_lib::io::output::Output::new(&mut stdout);
    solution::run(input, output);
}