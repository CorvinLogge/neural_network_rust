#[macro_export]
macro_rules! debug_only {
    ($expr:expr) => {
        let mut debug;

        unsafe { debug = DEBUG }

        if debug {
            $expr;
        }
    };

    ($block:block) => {
        let debug;

        unsafe { debug = DEBUG }

        if debug {
            $block;
        }
    };
}
