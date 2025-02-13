use crate::error::Error;
use crate::optimizers::Optimizer::{Adam, RmsProp, Sgd, SgdM};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum Optimizer {
    #[serde(alias = "sgd")] Sgd = 0,
    #[serde(alias = "sgdm")] SgdM = 1,
    #[serde(alias = "rms_prop")] RmsProp = 2,
    #[serde(alias = "adam")] Adam = 3,
}

impl TryFrom<u8> for Optimizer {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Sgd),
            1 => Ok(SgdM),
            2 => Ok(RmsProp),
            3 => Ok(Adam),
            _ => Err(Error::unknown_variant(vec![0, 1, 2, 3], value)),
        }
    }
}