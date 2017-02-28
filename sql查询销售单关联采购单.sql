select cheku_spec_order.create_time,cheku_spec_order.order_info,
	   cheku_spec_order.username,cheku_spec_order.address,cheku_spec_order.msrp,
	   cheku_spec_order.buy_num,cheku_spec_order.costs,cheku_spec_order.order_status,
       cheku_procurement_order.supplyGoods as supply,cheku_procurement_order.username as supplyname,cheku_procurement_order.goodsCity as fromcity
from chekumaster.cheku_spec_order right join chekumaster.cheku_procurement_order
ON  cheku_spec_order.proIds = cheku_procurement_order.id
where cheku_spec_order.transport_method = 2 and cheku_spec_order.del_flag != 1;