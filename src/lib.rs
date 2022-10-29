use std::ops::{Deref, DerefMut};
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::Arc;
use rand::Rng;

/// 一个位图用完标志
const ONE_BIT_MAP_FULL_FLAG: u64 = u64::MAX;

/// 位图组数量
const BIT_MAP_GROUP_COUNT: usize = 64;

pub trait PoolElementImpl: ElementImpl + Default + Sync + Send + 'static {}
impl<T> PoolElementImpl for T where T: ElementImpl + Default + Sync + Send + 'static {}

pub trait ElementImpl {
    fn reset(&mut self);
}

impl ElementImpl for Vec<u8> {
    fn reset(&mut self) {
        self.clear();
    }
}

pub struct PoolManager<T: PoolElementImpl> {
    pool: Arc<BufferPool<T>>,
}

impl<T: PoolElementImpl> PoolManager<T> {
    pub fn new(capacity: usize) -> Result<PoolManager<T>, String> {
        if capacity % BIT_MAP_GROUP_COUNT != 0 {
            return Err(format!(
                "capacity must be divisible by {}",
                BIT_MAP_GROUP_COUNT
            ));
        }

        let pool = Arc::new(BufferPool::new(capacity));

        Ok(PoolManager { pool })
    }

    /// 尝试从缓存获取
    pub fn try_get_from_cache(&self) -> Option<Cache<T>> {
        if let Some((data, index)) = self.pool.try_get_cache() {
            return Some(Cache::new(data, index, Some(self.pool.clone())));
        }

        return None;
    }

    /// 从缓存获取，如果缓存为空则创建(创建后的对象不会纳入缓存池)
    pub fn get_default_new(&self) -> Cache<T> {
        return match self.try_get_from_cache() {
            Some(data) => data,
            None => Cache::new(Box::new(T::default()), 0, None),
        };
    }
}

struct BufferPool<T: PoolElementImpl> {
    pool: Vec<AtomicPtr<T>>,
    _capacity: usize,
    bit_map: Vec<AtomicU64>,
}

impl<T: PoolElementImpl> BufferPool<T> {
    fn new(capacity: usize) -> BufferPool<T> {
        let bit_map_length = capacity / BIT_MAP_GROUP_COUNT;
        let mut bit_map_array = Vec::with_capacity(bit_map_length);
        for _ in 0..bit_map_length {
            bit_map_array.push(AtomicU64::new(0))
        }

        let mut pool = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            pool.push(AtomicPtr::new(ptr::null_mut::<T>()));
        }
        BufferPool {
            pool,
            _capacity: capacity,
            bit_map: bit_map_array,
        }
    }

    /// 尝试获取缓存
    fn try_get_cache(&self) ->Option<(Box<T>, usize)>{
        let mut rng = rand::thread_rng();
        let len = self.bit_map.len();
        let index:usize = rng.gen_range(0..len);
        for i in 0..len{
            let temp = (i + index)%len;
            if let Some(index) = self.mark_available_position(temp) {
                return Some((self.get_cache_by_position(index), index));
            }
        }

        None
    }

    /// 标记可用的位置索引
    fn mark_available_position(&self, index: usize) -> Option<usize> {
        let bit_data = &self.bit_map[index];
        loop {
            let bit = bit_data.load(Ordering::SeqCst);
            if bit >= ONE_BIT_MAP_FULL_FLAG {
                return None;
            }

            let mut i: usize = 8;
            while i > 0 {
                let temp = (bit >> ((i - 1) * 8)) & 255;
                if temp != 255 {
                    let mut j: usize = 8;
                    let mut find_index = 0;
                    while j > 0 {
                        if (temp >> (j - 1)) & 1 == 0 {
                            find_index = (i - 1) * 8 + j;
                            break;
                        }

                        j -= 1;
                    }

                    let new = bit | (1 << (find_index - 1));
                    if let Ok(_) =
                    bit_data.compare_exchange(bit, new, Ordering::SeqCst, Ordering::SeqCst)
                    {
                        return Some(
                            index * BIT_MAP_GROUP_COUNT + BIT_MAP_GROUP_COUNT - find_index,
                        );
                    }

                    break;
                }

                i -= 1;
            }
        }
    }

    /// 根据位置获取cache数据
    fn get_cache_by_position(&self, index: usize) -> Box<T> {
        let data_ptr = &self.pool[index];
        let ptr;
        if data_ptr.load(Ordering::Relaxed) == ptr::null_mut::<T>() {
            ptr = Box::into_raw(Box::new(T::default()));
        } else {
            ptr = data_ptr.swap(ptr::null_mut(), Ordering::SeqCst);
        }

        unsafe { Box::from_raw(ptr) }
    }

    /// 回收
    fn recovery_elem(&self, elem: Box<T>, index: usize) {
        let ptr = &self.pool[index];
        // 先将数据放入池中
        ptr.store(Box::into_raw(elem), Ordering::SeqCst);

        let bit_index = index / BIT_MAP_GROUP_COUNT;
        let bit_mark_index = index % BIT_MAP_GROUP_COUNT;
        let bit_map = &self.bit_map[bit_index];

        // 再修改可用标记
        loop {
            let bit_data = bit_map.load(Ordering::Relaxed);
            let new_data = bit_data ^ (1 << (BIT_MAP_GROUP_COUNT - bit_mark_index - 1));
            if let Ok(_) =
            bit_map.compare_exchange(bit_data, new_data, Ordering::SeqCst, Ordering::SeqCst)
            {
                return;
            }
        }
    }
}

pub struct Cache<T: PoolElementImpl> {
    inner: Option<Box<T>>,
    index: usize,
    pool: Option<Arc<BufferPool<T>>>,
}
impl<T: PoolElementImpl> Cache<T> {
    fn new(inner: Box<T>, index: usize, pool: Option<Arc<BufferPool<T>>>) -> Self {
        Cache {
            inner: Some(inner),
            index,
            pool,
        }
    }
}

impl<T: PoolElementImpl> Deref for Cache<T> {
    type Target = Box<T>;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref().unwrap()
    }
}

impl<T: PoolElementImpl> DerefMut for Cache<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.as_mut().unwrap()
    }
}

impl<T: PoolElementImpl> Drop for Cache<T> {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.as_ref() {
            let mut data = self.inner.take().unwrap();

            // 重置缓存
            data.deref_mut().reset();

            // 回收数据
            pool.recovery_elem(data, self.index);
        }
    }
}