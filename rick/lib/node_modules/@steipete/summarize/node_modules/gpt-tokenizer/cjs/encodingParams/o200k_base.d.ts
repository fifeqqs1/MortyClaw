import type { RawBytePairRanks } from '../BytePairEncodingCore.js';
import type { EncodingParams } from '../modelParams.js';
export declare const createO200KSpecialTokenMap: () => Map<string, number>;
export declare function O200KBase(bytePairRankDecoder: RawBytePairRanks): EncodingParams;
