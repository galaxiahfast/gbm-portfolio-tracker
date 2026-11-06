from scripts.near_miss_quote_audit import QuotePath
from scripts.near_miss_quote_fast import consume
from tests.test_near_miss_quotes import contract,quote


def test_stop_prefix_preserves_exit_and_omits_post_exit_gain():
    paths={h:QuotePath(contract()) for h in ('60','1440')}
    done={}
    assert not consume(paths,done,quote('2025-07-29T14:00:00Z'))
    assert not consume(paths,done,quote('2025-07-29T14:05:00Z'))
    assert consume(paths,done,quote('2025-07-29T14:05:30Z',97.,97.10))
    consume(paths,done,quote('2025-07-29T14:05:50Z',104.,104.10))
    for result in done.values():
        assert result['gross_quote_pnl_per_share']==-3.
        assert result['mfe_until_exit_per_share']==0.
        assert result['mfe_full_horizon_per_share'] is None


def test_same_population_terminal_deadline_not_selected_by_profit():
    paths={'60':QuotePath(contract())};done={}
    consume(paths,done,quote('2025-07-29T14:00:00Z'))
    consume(paths,done,quote('2025-07-29T14:05:00Z'))
    consume(paths,done,quote('2025-07-29T14:05:59Z',100.50,100.60))
    assert consume(paths,done,quote('2025-07-29T14:06:00Z',100.40,100.50))
    assert done['60']['exit']['outcome']=='TIMEOUT'
    assert done['60']['mfe_full_horizon_per_share'] is None
